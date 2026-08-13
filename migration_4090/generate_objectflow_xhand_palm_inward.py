#!/usr/bin/env python3
"""ObjectFlow XHand candidates with explicit inward palm orientation.

This is a versioned companion to generate_objectflow_xhand_poses.py. Existing
TCP-only pose banks are never modified. For each sample, the left TCP local
+X axis is aimed at the object center and the right TCP local -X axis is aimed
at the object center. A position+orientation Pyroki pose cost then solves the
full fixed-waist Tianji configuration.
"""
import argparse, csv, json, os, sys, copy
from pathlib import Path
import numpy as np, torch, trimesh
import jax.numpy as jnp, jaxlie, jaxls, pyroki as pk

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from utils.pyroki_ik import PyrokiRetarget

IK_URDF=Path(os.environ.get('XHAND_FULLBODY_IK_URDF', ROOT/'migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1_fixed_ik.urdf'))
ASSET_ROOT=Path(os.environ.get('OBJECTFLOW_ASSET_ROOT', ROOT/'external_assets/objectflow_20260812/ObjectFlow_3D_sim_assets_20260812'))

def rot_from_inward(inward, palm_tilt_deg=20.0, side='left', x_bias=-0.08):
    """Build a TCP frame matching the validated XHand convention.

    In the validated sphere grasp, *both* TCP local +X axes point inward:
    left +X toward world -Y, right +X toward world +Y. The old prototype
    incorrectly negated right +X and produced 45--122 degree orientation
    errors. A modest upward tilt reproduces the successful grasp family.
    """
    inward=np.asarray(inward,np.float32); inward[2]=0.0; inward/=max(np.linalg.norm(inward),1e-8)
    # Preserve the small x bias observed in the validated fixed-waist branch.
    h=np.array([x_bias, inward[1], 0.0],np.float32); h/=max(np.linalg.norm(h),1e-8)
    a=np.deg2rad(float(palm_tilt_deg)); x=np.cos(a)*h + np.array([0.,0.,np.sin(a)],np.float32); x/=max(np.linalg.norm(x),1e-8)
    z=np.array([0.,0.,-1.],np.float32)-np.dot(np.array([0.,0.,-1.],np.float32),x)*x
    if np.linalg.norm(z)<1e-5:
        z=np.array([1.,0.,0.],np.float32)-x[0]*x
    z/=np.linalg.norm(z)
    y=np.cross(z,x); y/=max(np.linalg.norm(y),1e-8)
    z=np.cross(x,y); z/=max(np.linalg.norm(z),1e-8)
    R=np.column_stack([x,y,z]).astype(np.float32)
    if np.linalg.det(R)<0: R[:,1]*=-1
    return jaxlie.SO3.from_matrix(jnp.asarray(R))

def solve_pose(robot, initial_q, targets, links, ori_weight, posture_q=None, posture_weight=0.0, hand_mask=None):
    joint_var=robot.joint_var_cls(0)
    factors=[pk.costs.pose_cost_analytic_jac(robot,joint_var,targets[i],jnp.asarray(links[i]),pos_weight=10.0,ori_weight=float(ori_weight)) for i in range(2)]
    factors.append(pk.costs.limit_cost(robot,joint_var,weight=10.0))
    if posture_q is not None and posture_weight > 0:
        w=np.full(len(initial_q), 0.02, dtype=np.float32)
        if hand_mask is not None: w[np.asarray(hand_mask,dtype=bool)] = float(posture_weight)
        factors.append(pk.costs.rest_cost(joint_var, jnp.asarray(posture_q), jnp.asarray(w)))
    problem=jaxls.LeastSquaresProblem(factors,[joint_var]).analyze()
    sol=problem.solve(initial_vals=jaxls.VarValues.make([joint_var.with_value(jnp.asarray(initial_q))]),linear_solver='dense_cholesky',verbose=False,termination=jaxls.TerminationConfig(max_iterations=128,early_termination=False),trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0))
    return np.asarray(sol[joint_var])

def load_pose(path,index):
    rows=list(csv.DictReader(path.open())); row=rows[max(0,min(index,len(rows)-1))]
    T=np.asarray([[float(row[f'm{i}{j}']) for j in range(4)] for i in range(4)],np.float32)
    return T,row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--episode',required=True); ap.add_argument('--frames',type=int,default=1); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--object-x',type=float,default=.50); ap.add_argument('--object-y',type=float,default=0.0); ap.add_argument('--object-z',type=float,default=.86)
    ap.add_argument('--wrist-side',type=float,default=.045); ap.add_argument('--tcp-inset',type=float,default=0.0); ap.add_argument('--wrist-z-offset',type=float,default=0.0)
    ap.add_argument('--orientation-weight',type=float,default=.30); ap.add_argument('--palm-tilt-deg',type=float,default=20.0); ap.add_argument('--hand-posture-template',type=Path,default=None); ap.add_argument('--template-index',type=int,default=0); ap.add_argument('--hand-posture-weight',type=float,default=0.0)
    args=ap.parse_args(); ep=ASSET_ROOT/args.episode; mesh_path=ep/'model.obj'; meta=json.loads((ep/'metadata.json').read_text())
    mesh=trimesh.load_mesh(mesh_path,force='mesh',process=False)
    if not isinstance(mesh,trimesh.Trimesh): mesh=trimesh.util.concatenate(tuple(mesh.geometry.values()))
    pose_path=ep/'pose_objectflow_interval.csv'; rows=list(csv.DictReader(pose_path.open())); indices=np.linspace(0,len(rows)-1,max(1,args.frames),dtype=int)
    solver=PyrokiRetarget(str(IK_URDF),['L_tcp','R_tcp']); robot=solver.robot; links=[robot.links.names.index(x) for x in ('L_tcp','R_tcp')]; names=list(robot.joints.actuated_names)
    posture=None
    if args.hand_posture_template:
        p=torch.load(args.hand_posture_template,map_location='cpu',weights_only=False)['samples'][args.template_index]; posture=dict(zip(p['joint_names'],p['full_body_q']))
    samples=[]
    for idx in indices:
        pose,row=load_pose(pose_path,int(idx)); cv=np.asarray([[1,0,0],[0,0,-1],[0,1,0]],np.float32); rot=cv@pose[:3,:3]; trans=cv@pose[:3,3]
        local=mesh.copy(); source_T=np.block([[rot,trans[:,None]],[np.zeros((1,3)),np.ones((1,1))]]); local.apply_transform(source_T); center0=local.bounding_box.centroid; delta=np.array([args.object_x,args.object_y,args.object_z],np.float32)-center0; local.apply_translation(delta); trans=trans+delta
        center=local.bounds.mean(0); side=max(.02,float(local.extents[1]*.5)+args.wrist_side-args.tcp_inset); z=float(center[2]+args.wrist_z_offset)
        positions=np.asarray([[center[0],center[1]+side,z],[center[0],center[1]-side,z]],np.float32)
        inward=np.asarray([center-positions[0],center-positions[1]],np.float32); targets=[jaxlie.SE3.from_rotation_and_translation(rot_from_inward(inward[i],palm_tilt_deg=args.palm_tilt_deg),jnp.asarray(positions[i])) for i in range(2)]
        q0=np.zeros(len(names),np.float32)
        if posture:
            q0=np.asarray([posture.get(n,0.0) for n in names],np.float32)
        posture_q=None; hand_mask=np.array(['_hand_' in n for n in names],dtype=bool)
        if posture:
            posture_q=np.asarray([posture.get(n,0.0) for n in names],np.float32)
        solved=solve_pose(robot,q0,targets,links,args.orientation_weight,posture_q,args.hand_posture_weight,hand_mask); fk=robot.forward_kinematics(jnp.asarray(solved)); achieved=[]; poserr=[]; orierr=[]; achieved_R=[]
        for i,li in enumerate(links):
            actual=jaxlie.SE3(fk[li]); achieved.append(np.asarray(actual.translation()).tolist()); achieved_R.append(np.asarray(actual.rotation().as_matrix()).tolist()); poserr.append(float(jnp.linalg.norm(actual.translation()-targets[i].translation()))); orierr.append(float(jnp.linalg.norm((targets[i].rotation().inverse()@actual.rotation()).log())))
        # The posture template is an IK warm start only. Overwriting hand
        # joints after the pose solve would destroy the newly solved palm
        # orientation and make the recorded target/achieved frames stale.
        full_q=solved.copy()
        target_R=[np.asarray(t.rotation().as_matrix()).tolist() for t in targets]
        samples.append({'schema':'objectflow_scanned_xhand_palm_inward_pose_v1','episode':args.episode,'frame_index':int(idx),'source_frame':int(row['source_frame']),'object_mesh_path':str(mesh_path.resolve()),'object_pose_source_opencv':pose.tolist(),'object_pose_robot_preview':np.block([[rot,trans[:,None]],[np.zeros((1,3)),np.ones((1,1))]]).tolist(),'object_extents_m':meta['extents_m'],'target_tcp_positions_world':positions.tolist(),'target_tcp_rotations_world':target_R,'achieved_tcp_positions_world':achieved,'achieved_tcp_rotations_world':achieved_R,'full_body_ik_q':full_q.tolist(),'joint_names':names,'ik_position_error_m':poserr,'ik_orientation_error_rad':orierr,'palm_constraint':{'left_local_axis':'+X inward','right_local_axis':'+X inward','palm_tilt_deg':args.palm_tilt_deg,'orientation_weight':args.orientation_weight},'physical_validation_pending':True,'source_quality':meta})
    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save({'schema':'objectflow_scanned_xhand_palm_inward_pose_bank_v1','samples':samples},args.output); print(json.dumps({'output':str(args.output.resolve()),'samples':len(samples),'max_pos_err_m':max(max(s['ik_position_error_m']) for s in samples),'max_ori_err_deg':max(max(s['ik_orientation_error_rad']) for s in samples)*180/np.pi},indent=2))
if __name__=='__main__': main()
