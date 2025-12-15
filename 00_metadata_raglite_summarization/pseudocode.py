
'''
INGEST:
  taxonomy_records = load_taxonomy()
  example_records  = load_gold_examples()
  index = build_vector_index(taxonomy_records + example_records)

RUN(doc):
  text = extract_text(doc)
  text = clean(text)
  doc_type = classify_cheap(text)  # optional heuristic

  high_signal = pick_title_abstract_headings(text)
  query = high_signal

  retrieved_taxonomy = retrieve(index, query, filter="taxonomy", top_k=12)
  retrieved_examples = retrieve(index, query, filter="example",  top_k=3)

  prompt = build_prompt(
     doc_text=high_signal,
     taxonomy_options=retrieved_taxonomy,
     example_records=retrieved_examples,
     json_schema=METADATA_SCHEMA
  )

  raw = llm.generate(prompt)
  metadata = parse_json(raw)
  validate(metadata)               # schema + allowed taxonomy checks
  return metadata
'''