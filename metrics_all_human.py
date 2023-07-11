import argparse
import multiprocessing
import os
import sys

sys.path.insert(0, os.getcwd())
from eval_utils import ama_eval, format_latex_table


def main(opts):
    # Allow CUDA from within multiprocessing
    mp = multiprocessing.get_context("spawn")

    evals = [
        ("dasr_human", "human", "final", "params/params_latest.pth"),
    ]
    videos = ["T_samba1", "D_bouncing1", "D_handstand1"]

    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        num_gpus = int(os.popen("nvidia-smi -L | wc -l").read())
        gpus = list(range(num_gpus))
    else:
        gpus = [int(n) for n in os.getenv("CUDA_VISIBLE_DEVICES").split(",")]

    # Evaluate
    if opts.do_evaluate:
        procs = []
        for i, ev in enumerate(evals):
            print("Evaluating", i, ev)
            ev_label, ev_name, ev_detail, ev_params = ev[:4]
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpus[i % len(gpus)]} python3 main.py --eval "
                f"--load_params output/{ev_name}_{ev_detail}/{ev_params} --use_ddp False --use_cache_gt False "
                f"--save_pose_dfms_img False --save_pose_dfms_mesh True --vis_pose_dfms_mplex_3d False"
            )
            print(cmd)
            os.system(cmd)


    # Compute metrics
    metrics = []
    if opts.do_metrics:
        metrics += find_metrics_load_dirs(evals)
    if len(metrics) > 0:
        print(f"\tT_samba1\t\t\tD_bouncing1\t\t\tD_handstand1\t\t\t")
        print(f"seqname\tcd\tf@10cm\tf@5cm\tf@2cm\tcd\tf@10cm\tf@5cm\tf@2cm\tcd\tf@10cm\tf@5cm\tf@2cm")
        procs = []
        results = []

        # Spawn procs
        for i, metric in enumerate(metrics):
            if len(metric) == 2:
                label, load_dir = metric
                kwargs = {}
            elif len(metric) == 3:
                label, load_dir, kwargs = metric

            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])
            print("Computing metrics", i, label, load_dir, kwargs)

            eval_a = ama_eval(load_dir, "T_samba", 1, render_vid=load_dir, **kwargs)
            print(f"{label} T_samba", eval_a)
            eval_b = ama_eval(load_dir, "D_bouncing", 1, render_vid=load_dir, **kwargs)
            print(f"{label} D_bouncing", eval_b)
            eval_c = ama_eval(load_dir, "D_handstand", 1, render_vid=load_dir, **kwargs)
            print(f"{label} D_handstand", eval_c)

            cd_a, f010_a, f005_a, f002_a = eval_a
            cd_b, f010_b, f005_b, f002_b = eval_b
            cd_c, f010_c, f005_c, f002_c = eval_c
            print(f"{label}\t"
                  f"{100 * cd_a:.2f}\t{100 * f010_a:.1f}\t{100 * f005_a:.1f}\t{100 * f002_a:.1f}\t"
                  f"{100 * cd_b:.2f}\t{100 * f010_b:.1f}\t{100 * f005_b:.1f}\t{100 * f002_b:.1f}\t"
                  f"{100 * cd_c:.2f}\t{100 * f010_c:.1f}\t{100 * f005_c:.1f}\t{100 * f002_c:.1f}")
            
            results.append((i, label, load_dir, eval_a, eval_b, eval_c))

        # Print outputs
        latex_table = []
        for i, label, load_dir, eval_a, eval_b, eval_c in sorted(results):
            cd_a, f010_a, f005_a, f002_a = eval_a
            cd_b, f010_b, f005_b, f002_b = eval_b
            cd_c, f010_c, f005_c, f002_c = eval_c
            latex_table.append((
                label, cd_a, f010_a, f005_a, f002_a, cd_b, f010_b, f005_b, f002_b, cd_c, f010_c, f005_c, f002_c
            ))
            print(f"{label}\t"
                  f"{100 * cd_a:.2f}\t{100 * f010_a:.1f}\t{100 * f005_a:.1f}\t{100 * f002_a:.1f}\t"
                  f"{100 * cd_b:.2f}\t{100 * f010_b:.1f}\t{100 * f005_b:.1f}\t{100 * f002_b:.1f}\t"
                  f"{100 * cd_c:.2f}\t{100 * f010_c:.1f}\t{100 * f005_c:.1f}\t{100 * f002_c:.1f}")

        # Format latex table
        format_spec = [(None, 0, 0)] + 3 * ([(True, 100, 2)] + 3 * [(False, 100, 1)])
        print(format_latex_table(latex_table, format_spec))


def find_metrics_load_dirs(evals):
    """Find output indices of evals

    Args
        evals [List(Tuple(str, str))]: Input DASR evals

    Returns
        load_dirs [List(string)]: DASR metrics load_dirs
    """
    ls_out = os.listdir("output")
    metrics = []
    for ev in evals:
        if len(ev) == 4:
            ev_label, ev_name, ev_detail, ev_params = ev
        elif len(ev) == 5:
            ev_label, ev_name, ev_detail, ev_params, ev_kwargs = ev
        
        idx = len([x for x in ls_out if ev_name in x and ev_detail in x]) - 1
        load_dir = f"output/{ev_name}_{ev_detail}_{idx}/eval_pose_dfms_mesh_000"

        if len(ev) == 4:
            metrics.append((ev_label, load_dir))
        elif len(ev) == 5:
            metrics.append((ev_label, load_dir, ev_kwargs))

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--do_evaluate", type=str, choices=["True", "False"], default="True")
    parser.add_argument("--do_metrics", type=str, choices=["True", "False"], default="True")

    opts = parser.parse_args()

    for attr in opts.__dict__:
        if getattr(opts, attr) == "True":
            setattr(opts, attr, True)
        elif getattr(opts, attr) == "False":
            setattr(opts, attr, False)

    return opts


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
