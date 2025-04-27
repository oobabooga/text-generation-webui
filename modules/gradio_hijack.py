'''
Most of the code here was adapted from:
https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14184
'''

import inspect
import warnings
from functools import wraps

import gradio as gr
import gradio.routes
import gradio.utils
from starlette.middleware.trustedhost import TrustedHostMiddleware

from modules import shared

orig_create_app = gradio.routes.App.create_app


# Be strict about only approving access to localhost by default
def create_app_with_trustedhost(*args, **kwargs):
    app = orig_create_app(*args, **kwargs)

    if not (shared.args.listen or shared.args.share):
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1"]
        )

    return app


gradio.routes.App.create_app = create_app_with_trustedhost
gradio.utils.launch_counter = lambda: None


class GradioDeprecationWarning(DeprecationWarning):
    pass


def repair(grclass):
    if not getattr(grclass, 'EVENTS', None):
        return

    @wraps(grclass.__init__)
    def __repaired_init__(self, *args, tooltip=None, source=None, original=grclass.__init__, **kwargs):
        if source:
            kwargs["sources"] = [source]

        allowed_kwargs = inspect.signature(original).parameters
        fixed_kwargs = {}
        for k, v in kwargs.items():
            if k in allowed_kwargs:
                fixed_kwargs[k] = v
            else:
                warnings.warn(f"unexpected argument for {grclass.__name__}: {k}", GradioDeprecationWarning, stacklevel=2)

        original(self, *args, **fixed_kwargs)

        self.webui_tooltip = tooltip

        for event in self.EVENTS:
            replaced_event = getattr(self, str(event))

            def fun(*xargs, _js=None, replaced_event=replaced_event, **xkwargs):
                if _js:
                    xkwargs['js'] = _js

                return replaced_event(*xargs, **xkwargs)

            setattr(self, str(event), fun)

    grclass.__init__ = __repaired_init__
    grclass.update = gr.update


for component in set(gr.components.__all__ + gr.layouts.__all__):
    repair(getattr(gr, component, None))


class Dependency(gr.events.Dependency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def then(*xargs, _js=None, **xkwargs):
            if _js:
                xkwargs['js'] = _js

            return original_then(*xargs, **xkwargs)

        original_then = self.then
        self.then = then


gr.events.Dependency = Dependency

gr.Box = gr.Group
