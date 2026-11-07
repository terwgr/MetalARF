"""The cross-process GPU lock the NWP suite shares (:mod:`woof.mcp.gpulock`).

Recast WOOF carries no MCP server: this package keeps only the lock, which
``woof suite`` (woof.nwpsuite: ledger, retention, run, worker) takes before it
starts work on a card.
"""
