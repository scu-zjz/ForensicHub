_lazy_model_map = {
    "CAFTB_Net": "ForensicHub.tasks.document.models.caftb_net.caftb_net",
    "DTD": "ForensicHub.tasks.document.models.dtd.dtd",
    "FFDN": "ForensicHub.tasks.document.models.ffdn.ffdn",
    "PSNet": "ForensicHub.tasks.document.models.psnet.psnet",
    "Tifdm": "ForensicHub.tasks.document.models.tifdm.tifdm",
}

_lazy_postfunc_map = {
    "dtd_post_func": "ForensicHub.tasks.document.models.dtd.dtd_post_function"
}

from .dtd.dtd_post_function import dtd_post_func