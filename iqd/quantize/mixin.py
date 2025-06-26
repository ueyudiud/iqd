import torch
from diffusers import ConfigMixin
from diffusers import ModelMixin
from diffusers.pipelines.pipeline_loading_utils import LOADABLE_CLASSES, ALL_IMPORTABLE_CLASSES
from diffusers.utils import PushToHubMixin
from huggingface_hub.utils import validate_hf_hub_args

from iqd.utils.load import load_class
from .module.modeling_utils import transform_submodules

QUANTIZER_CONFIG_NAME = "quantizer.json"

LOADABLE_CLASSES['iqd'] = {
    "Quantizer": ["save_pretrained", "from_pretrained"],
}

ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES['iqd'])

class QuantizerMixin(ConfigMixin, PushToHubMixin):
    config_name = QUANTIZER_CONFIG_NAME

    def _attach_to(self, module):
        from iqd.utils.load import fetch_class_library_tuple

        module._quantizer = self
        module.register_to_config(_quantizer=fetch_class_library_tuple(self))

    def convert(self, model, train=True):
        raise NotImplemented

    def quantize_pretrained(self, model_path: str, **kwargs):
        raise NotImplemented

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path=None, return_unused_kwargs=False, **kwargs):
        config, kwargs = cls.load_config(
            pretrained_model_name_or_path,
            return_unused_kwargs=True,
            **kwargs)

        return cls.from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)

class QModelMixin(ModelMixin, ConfigMixin):
    @classmethod
    def convert_pretrained(cls, origin_model_name_or_path, quantizer, **kwargs):
        model = cls.from_pretrained(origin_model_name_or_path, **kwargs, _load_origin=True)
        quantizer.convert(model, True)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, quantizer=None, **kwargs):
        if quantizer is None and not kwargs.get('_load_origin', False):
            quantizer = QuantizerMixin.load_config(pretrained_model_name_or_path, False, False, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, quantizer=quantizer, **kwargs)

    @classmethod
    def from_config(cls, config=None, return_unused_kwargs=False, **kwargs):
        if kwargs.pop('_load_origin', False):
            return super().from_config(config, return_unused_kwargs, **kwargs)

        quantizer = kwargs.pop('quantizer')
        device = kwargs.pop('device', None) or torch.get_default_device()
        if not isinstance(quantizer, QuantizerMixin):
            quantizer_library, quantizer_class_name = config['_quantizer']
            quantizer_class = load_class(quantizer_library, quantizer_class_name)
            if not issubclass(quantizer_class, QuantizerMixin):
                raise ValueError(f"{quantizer_class} is not a valid quantizer class.")
            quantizer = quantizer_class.from_config(quantizer)
        quant_train = config.get('quant_train', False)

        model, kwargs = super().from_config(config, True, device='meta', **kwargs)
        model = quantizer.convert(model, quant_train)
        if device.type != 'meta':
            model.to_empty(device=device)

        return (model, kwargs) if return_unused_kwargs else model

    def save_pretrained(self, save_directory, is_main_process=True, save_function=None, safe_serialization=True,
                        variant=None, max_shard_size="10GB", push_to_hub=False, **kwargs):
        if hasattr(self, '_quantizer'):
            quantizer = self._quantizer
            assert isinstance(quantizer, QuantizerMixin)
            quantizer.save_pretrained(save_directory, push_to_hub, **kwargs)
        super().save_pretrained(save_directory, is_main_process, save_function, safe_serialization, variant, max_shard_size, push_to_hub, **kwargs)

    def detach_(self):
        from .module.module import ModuleQat
        transform_submodules(lambda _, m: m.detach() if isinstance(m, ModuleQat) else None, self)
        return self