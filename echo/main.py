import sys
import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import initialize_ray
from skyrl.train.entrypoints import BasePPOExp, validate_cfg
from skyrl.backends.skyrl_train.utils.ppo_utils import register_policy_loss


@register_policy_loss("echo_grpo")
def compute_echo_grpo_loss():
    pass


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    exp = BasePPOExp(cfg)
    exp.run()


def main():
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
