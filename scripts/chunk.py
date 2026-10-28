from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, TokenTextSplitter
from transformers import AutoTokenizer

from rago.utils import PATH_DEFAULT_DATA_DIR
from scripts.utils import load_docs_from_jsonl, save_docs_to_jsonl

CHUNKS_TO_REMOVE = ["## (PAGE VIERGE)", "(PAGE VIERGE)", "## INTENTIONNELLEMENT BLANC", "<!-- image -->"]
def aggregate(tokenizer: AutoTokenizer, texts: list[Document], max_num_tokens: int = 1024) -> list[Document]:
    aggregated_texts: list[Document] = []
    prev_doc_size = 0
    nb_tokens = [len(tokens) for tokens in tokenizer([t.page_content for t in texts], add_special_tokens=False)["input_ids"]]
    for t, t_nb_tokens in zip(texts, nb_tokens):
        if len(aggregated_texts) == 0 or (t_nb_tokens+prev_doc_size > max_num_tokens):
            aggregated_texts.append(t)
            prev_doc_size = t_nb_tokens
            continue
        aggregated_texts[-1].page_content += "\n"+ t.page_content
        prev_doc_size += t_nb_tokens
    return aggregated_texts

if __name__ == "__main__":
    path_data = Path(PATH_DEFAULT_DATA_DIR)
    path_with_page = path_data / "eval_llm_2026_with_pages"
    path_split_documents = path_with_page/"documents"
    path_chunked_splits  = path_with_page/"chunks"
    path_chunked_splits.mkdir(exist_ok=True)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    max_num_tokens = 1024
    min_num_characters = 128
    markdown_splitter  = MarkdownHeaderTextSplitter(headers_to_split_on,strip_headers=False)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    token_splitter = TokenTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer, chunk_size = max_num_tokens, chunk_overlap = 128)

    for fpath in path_split_documents.iterdir():
        pages = load_docs_from_jsonl(fpath)
        chunks: list[Document] = []
        collapse_previous_chunk = False
        for idx, p in enumerate(pages):

            if collapse_previous_chunk:
                last_chunk = chunks.pop(-1)
                p.page_content = last_chunk.page_content+ "\n" + p.page_content
                collapse_previous_chunk = False
            new_chunks = token_splitter.split_documents([p])
            if len(new_chunks) ==1:
                if new_chunks[0].page_content.strip() in CHUNKS_TO_REMOVE:
                    continue
                if len(new_chunks[0].page_content) < min_num_characters and idx < len(pages)-1:
                    collapse_previous_chunk = True
                if len(chunks) ==0 and idx == len(pages)-1 and len(new_chunks[0].page_content) < min_num_characters:
                    continue
                if len(new_chunks[0].page_content) < min_num_characters and idx == len(pages)-1:
                    chunks[-1].page_content += "\n"+new_chunks[0].page_content
                    continue

            chunks.extend(
            Document(
                page_content = t.page_content,
                metadata =  p.metadata | t.metadata ,
            ) for t in new_chunks
            )

        save_docs_to_jsonl(chunks, path_chunked_splits/fpath.name)
