import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")


with st.sidebar:
    st.link_button(
        "Github로 이동",
        url="https://github.com/2wonbin/nomad-site-gpt",
        help="해당 프로젝트의 깃허브 레포지토리로 이동합니다.",
        use_container_width=True,
    )
api_key = ""
with st.sidebar:
    api_key = st.text_input(
        "OpenAI API Key",
        placeholder="OpenAI API Key를 입력하세요",
    )
if api_key == "":
    st.error("OpenAI API Key를 입력하세요.")
    st.stop()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    openai_api_key=api_key,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
    Your turn!
    Question: {question}
    """
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            cite souces. DO NOT modify the source, keep it as a link.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    question = inputs["question"]
    answers = inputs["answers"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"Answer: {answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n" for answer in answers
    )
    return choose_chain.invoke({"question": question, "answers": condensed})


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_documents(docs, embedding=embeddings)

    return vector_store.as_retriever()


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        if message["message"] == "":
            continue
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


if "messages" not in st.session_state:
    st.session_state["messages"] = []


st.markdown(
    """
    # SiteGPT
            
    이 페이지는은 OpenAI의 모델을 사용하여 웹사이트의 Sitemap을 통해 정보를 얻을 수 있습니다.
    Sitemap URL을 입력하고 질문을 입력하면 해당 질문에 대한 답변을 얻을 수 있습니다.
"""
)


url = ""
if api_key != "":
    with st.sidebar:
        url = st.text_input(
            "정보를 얻고 싶은 사이트의 Sitemap URL을 입력하세요.",
            placeholder="https://example.com",
        )


if url != "":
    if ".xml" not in url:
        with st.sidebar:
            st.error("파일 형식이 올바르지 않습니다. xml 파일을 입력하세요.")
    else:
        retriever = load_website(url)

    send_message("사이트 분석이 끝났습니다. 궁금한게 있으신가요?", "ai", save=False)
    paint_history()

    user_question = st.chat_input("사이트에 대해 궁금한게 있으면 물어보세요.")
    if user_question:
        send_message(user_question, "human")

        chain = {
            "docs": retriever,
            "question": RunnablePassthrough(),
        } | RunnableLambda(get_answers)

        final_chain = chain | RunnableLambda(choose_answer)

        with st.spinner("답변을 찾는 중입니다..."):
            response = final_chain.invoke(user_question)
            result = response.content
            send_message(result, "ai")

else:
    st.error("Sitemap URL을 입력하세요.")
    st.session_state["messages"] = []
