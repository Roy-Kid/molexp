import hashlib
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from .models import AssetMeta

class AssetManager:
    def __init__(self, root_path: str):
        self.root = Path(root_path).resolve()
        self.assets_dir = self.root / "assets"
        self.objects_dir = self.assets_dir / "objects"
        self.index_dir = self.assets_dir / "index"

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def ingest_file(self, file_path: str, producer_run_id: Optional[str] = None, 
                    tags: List[str] = [], move: bool = False) -> AssetMeta:
        """
        Ingest a file into the asset repository.
        If move=True, the original file is moved; otherwise copied.
        """
        src_path = Path(file_path).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"File {file_path} not found.")

        # Calculate hash
        file_hash = self._calculate_hash(src_path)
        
        # Determine destination
        # Structure: objects/ab/ab1234...
        prefix = file_hash[:2]
        dest_dir = self.objects_dir / prefix
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file_hash

        # Check if already exists
        if not dest_path.exists():
            if move:
                shutil.move(str(src_path), str(dest_path))
            else:
                shutil.copy2(str(src_path), str(dest_path))
        
        # Create or Update Metadata
        # Note: In a real system, we might check if meta exists and just return it, 
        # but here we might want to update tags or producer if it's a new context.
        # For CAS, the ID is the hash.
        asset_id = file_hash # Simple strategy: ID = Hash
        
        meta_path = self.index_dir / f"{asset_id}.json"
        
        if meta_path.exists():
            # Already exists, maybe update tags? For now, just load and return
            with open(meta_path, "r") as f:
                data = json.load(f)
            return AssetMeta(**data)
        
        # Create new meta
        stat = dest_path.stat()
        meta = AssetMeta(
            id=asset_id,
            hash=file_hash,
            size_bytes=stat.st_size,
            mime_type="application/octet-stream", # TODO: guess mime type
            created_at=datetime.now(),
            producer_run_id=producer_run_id,
            tags=tags
        )
        
        with open(meta_path, "w") as f:
            f.write(meta.model_dump_json(indent=2))
            
        return meta

    def get_asset_path(self, asset_id: str) -> Path:
        """Resolve asset ID to physical path."""
        # Assuming ID is hash for now
        prefix = asset_id[:2]
        path = self.objects_dir / prefix / asset_id
        if not path.exists():
            raise FileNotFoundError(f"Asset {asset_id} not found in repository.")
        return path

    def get_asset_meta(self, asset_id: str) -> AssetMeta:
        meta_path = self.index_dir / f"{asset_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Asset metadata for {asset_id} not found.")
        
        with open(meta_path, "r") as f:
            data = json.load(f)
        return AssetMeta(**data)
