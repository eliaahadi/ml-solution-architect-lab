from typing import Dict, Any, Set
from jsonschema import validate as js_validate
from jsonschema.exceptions import ValidationError
from .metadata_schema import METADATA_JSON_SCHEMA, MetadataRecord

def enforce_schema(obj: Dict[str, Any]) -> MetadataRecord:
    js_validate(instance=obj, schema=METADATA_JSON_SCHEMA)
    return MetadataRecord(**obj)

def enforce_allowed_taxonomy(obj: Dict[str, Any], allowed_ids: Set[str]) -> None:
    for field in ["topics", "programs", "geographies"]:
        for v in obj.get(field, []):
            if v not in allowed_ids:
                raise ValidationError(f"Value '{v}' in {field} not in allowed taxonomy set")
    sens = obj.get("sensitivity")
    if sens is not None and sens not in allowed_ids:
        raise ValidationError(f"sensitivity '{sens}' not in allowed taxonomy set")