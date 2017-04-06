import sys
import argparse
import os
from six.moves import shlex_quote

def new_cmd(session, name, cmd, mode, logdir, shell):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shl)
