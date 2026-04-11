"""molq submission plugin — provides cluster backends for ``molexp run``.

When molq is installed, ``--slurm``, ``--pbs``, and ``--lsf`` flags are
available on ``molexp run``.  If molq is not installed the import fails at
runtime with a clear error message.
"""
