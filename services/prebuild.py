import os

def prebuild_all_embeddings(company_store):
    col = company_store.col_enterprise
    for enterprise_doc in col.find({}):
        id_str = str(enterprise_doc.get("_id"))
        emb_fp = company_store._embedding_file_for(id_str)
        if not os.path.exists(emb_fp):
            company_store._build_and_store_embeddings_local(enterprise_doc, id_str)
        company_store.get_vector_store(id_str)