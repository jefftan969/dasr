import json
import sys

from data_utils import data_info_from_banmo_config

banmo_model = None

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with attribute syntax"""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]


def register_banmo(banmo_path, seqname):
    """Properly load in a banmo model.
    Save opts dict using the following code within banmo constructor:
    # import json
    # with open("{BANMO_PATH}/banmo_opts.json", "w"):
    #     json.dump({k: v.value for k, v in dict(opts._flags()).items()})

    Args
        banmo_path [str]: Path to banmo dependencies, including banmo_opts.json,
            params.pth, vars.npy, and mesh_rest.obj
        seqname [str]: Banmo seqname to use
    """
    with open(f"{banmo_path}/banmo_opts.json", "rb") as banmo_opts_file:
        banmo_opts = EasyDict(json.load(banmo_opts_file))
    banmo_opts.checkpoint_dir = "."
    banmo_opts.logname = ""
    banmo_opts.model_path = f"{banmo_path}/params.pth"
    banmo_opts.seqname = seqname
    banmo_data_info = data_info_from_banmo_config(seqname)

    sys.path.append(f"{banmo_path}/src")
    from nnutils.train_utils import v2s_trainer
    from nnutils.geom_utils import extract_mesh
    trainer = v2s_trainer(banmo_opts, is_eval=True)
    trainer.define_model(banmo_data_info)
    global banmo_model
    banmo_model = trainer.model


def banmo():
    """Return the global banmo model loaded in with register_banmo()"""
    return banmo_model
