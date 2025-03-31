import yaml
import copy
import argparse
import json

class Struct:
    def __init__(self, override=None, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Struct(override=override, **value)
            else:
                if override is None:
                    self.__dict__[key] = value
                else:
                    self.__dict__[key] = override[key] if (key in override and override[key] is not None) else value

    def to_dict(self):
        # recursively convert to dict
        return {
            k: v.to_dict() if isinstance(v, Struct) else v
            for k, v in self.__dict__.items()
        }

    def __getitem__(self, index):
        return self.__dict__[index]

class Config:
    def __init__(self, config_file=None, **kwargs):
        assert config_file is not None
        
        _config = load_config(path=config_file)
        # override config file with argparse args if not None
        for key, value in _config.items():
            if isinstance(value, dict):
                self.__dict__[key] = Struct(override=kwargs, **value)
            else:
                self.__dict__[key] = kwargs[key] if (key in kwargs and kwargs[key] is not None) else value

    def __getitem__(self, index):
        return self.__dict__[index]

    def get_vae_beta(self):
        self.vae_beta_val = beta

    def to_dict(self):
        # recursively convert to dict
        return {
            k: v.to_dict() if isinstance(v, Struct) else v
            for k, v in self.__dict__.items()
        }

    def save2yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def __str__(self):
        def prepare_dict4print(dict_):
            tmp_dict = copy.deepcopy(dict_)

            def recursive_change_list_to_string(d, summarize=16):
                for k, v in d.items():
                    if isinstance(v, dict):
                        recursive_change_list_to_string(v)
                    elif isinstance(v, list):
                        d[k] = (
                            (
                                str(
                                    v[: summarize // 2] + ["..."] + v[-summarize // 2 :]
                                )
                                + f" (len={len(v)})"
                            )
                            if len(v) > summarize
                            else str(v) + f" (len={len(v)})"
                        )
                    else:
                        pass

            recursive_change_list_to_string(tmp_dict)
            return tmp_dict

        return json.dumps(prepare_dict4print(self.to_dict()), indent=4, sort_keys=False)

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # run script to debug config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/debug.yaml",
    )
    parser.add_argument('--save', action='store_true', help='save model and configs to file (default: False)')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    _args = parser.parse_args()
    cfg = Config(**_args.__dict__)

    # TMP
    print("Save:", cfg.save)
    cfg.save_path = 'what'
    print("Save_path:", cfg.save_path is None)
    print(cfg.save_path)
    print(f"The config of this process is:\n{cfg}")