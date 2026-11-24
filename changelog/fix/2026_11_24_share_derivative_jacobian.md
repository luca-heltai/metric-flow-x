PR: pending
- Reuse the assembled state-independent derivative Jacobian across immutable solver snapshots, while keeping prior matrix versions intact when setup or mass data invalidates the cache.
