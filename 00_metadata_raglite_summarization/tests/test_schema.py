from metaragl.schema.validate import enforce_schema

def test_schema_ok():
    obj = {
      "title":"t","doc_type":"memo","abstract":"a",
      "topics":[],"programs":[],"geographies":[],
      "sensitivity":None,"keywords":[],"evidence_snippets":[]
    }
    rec = enforce_schema(obj)
    assert rec.title == "t"