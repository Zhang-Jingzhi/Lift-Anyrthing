#!/usr/bin/env python3
"""Create per-hand finger-closure variants from one XHand full-body pose."""
import argparse, copy
from pathlib import Path
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--sample-index', type=int, default=6)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--left-values', nargs='+', type=float, required=True)
    ap.add_argument('--right-values', nargs='+', type=float, required=True)
    args = ap.parse_args()
    payload = torch.load(args.dataset, map_location='cpu')
    base = payload['samples'][args.sample_index]
    out = []
    # All non-bend finger joints share the nominal closure value in the
    # generator; keep the small index-bend coupling at its source value.
    for lv in args.left_values:
        for rv in args.right_values:
            s = copy.deepcopy(base)
            q = list(s['full_body_q'])
            for i, n in enumerate(s['joint_names']):
                if n.startswith('left_hand_') and n.endswith(('joint1','joint2')):
                    q[i] = float(lv)
                elif n.startswith('right_hand_') and n.endswith(('joint1','joint2')):
                    q[i] = float(rv)
            s['full_body_q'] = q
            s['optimization_candidate'] = dict(s.get('optimization_candidate', {}), left_closure_value=float(lv), right_closure_value=float(rv))
            s['geometry_pass'] = False
            s['isaaclab_physical_validated'] = False
            out.append(s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'schema':'xhand_fullbody_grasp_pose_handclosure_search_v1','samples':out}, args.output)
    print({'output':str(args.output),'count':len(out),'labels':[s['optimization_candidate'] for s in out]})

if __name__ == '__main__': main()
