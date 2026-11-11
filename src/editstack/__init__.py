"""editstack: evidence-driven hand-editing of model weights. One stack = one editable model.

Python API (inside a stack folder, or with EDITSTACK_ROOT set):
    from editstack import Stack, ops, fit
    st = Stack.find()
    a = st.weights("my_variant")        # original + edit chain, in memory (a.tensors, a.changed, a.stats)
    policy = st.policy("my_variant")    # takes a RAM slot, builds the runnable model via env/adapter.py
"""
from .stack import Stack, StackError  # noqa: F401
from . import ops, fit  # noqa: F401

__version__ = "0.1.0"
