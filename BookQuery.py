import os
import pickle
from io import BytesIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from PyPDF2 import PdfFileReader, PdfFileWriter
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from channel import channel_factory
from common.log import logger
from config import conf, load_config
from plugins import *
import faiss
import signal
import sys


OPENAI_API_KEY = "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"

index_name = "langchain-finance-class-for-leaders.index"
namespace = "faiss-book.pkl"

def isExistTrainFile():
    return os.path.exists(index_name)

def extract_text_from_pdf(pdf_path):
    output = BytesIO()
    with open(pdf_path, "rb") as f:
        parser = PDFParser(f)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
    text = output.getvalue().decode("utf-8")
    output.close()
    return text

#WeChat Bot portion
def sigterm_handler_wrap(_signo):
    old_handler = signal.getsignal(_signo)

    def func(_signo, _stack_frame):
        logger.info("signal {} received, exiting...".format(_signo))
        conf().save_user_datas()
        if callable(old_handler):  #  check old_handler
            return old_handler(_signo, _stack_frame)
        sys.exit(0)

    signal.signal(_signo, func)

def split_pdf(pdf_path, output_dir):
    with open(pdf_path, "rb") as f:
        pdf = PdfFileReader(f)
        total_pages = pdf.getNumPages()
        chapters = []
        current_chapter = []
        
        for page_num in range(total_pages):
            text = pdf.getPage(page_num).extract_text()
            if "Chapter" in text:
                if current_chapter:
                    chapters.append(current_chapter)
                current_chapter = []
            current_chapter.append(page_num)
        
        if current_chapter:
            chapters.append(current_chapter)
        
        file_count = 1
        for chapter in chapters:
            start_page = chapter[0]
            end_page = chapter[-1] + 1
            output_path = os.path.join(output_dir, f"part-{file_count}.pdf")
            
            with open(output_path, "wb") as output_file:
                writer = PdfFileWriter()
                for page_num in range(start_page, end_page):
                    writer.addPage(pdf.getPage(page_num))
                writer.write(output_file)
            
            file_count += 1

def train():
    # Define the paths to the PDF files
    pdf_paths = [
        "data/领导者的极简财务课.pdf",
        "data/财报思维-写给忙碌者的财报学习书5.6.pdf",
        "data/《吞噬金钱，看透财务造假》-作者王峰-20220116-改茶楼.pdf",
        "data/《财务那点事儿-内文出片 - 副本.pdf",
        # Add more file paths as needed
    ]

    # Split the books into smaller PDFs
    output_dir = "data/split_pdfs"
    os.makedirs(output_dir, exist_ok=True)
    for pdf_path in pdf_paths:
        split_pdf(pdf_path, output_dir)

    # Load the split PDFs
    pdf_paths = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))

    context_windows = []
    window_size = 2000  # Adjust the window size as needed
    overlap = 500  # Adjust the overlap size as needed

    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        sentences = text.strip().split('\n')
        for i in range(0, len(sentences), window_size - overlap):
            window_text = ' '.join(sentences[i:i+window_size])
            context_windows.append(window_text)

    print(f'Now you have {len(context_windows)} context windows')

    # Build Semantic Index
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    store = FAISS.from_texts(context_windows, embeddings)
    faiss.write_index(store.index, index_name)
    store.index = None

    with open(namespace, "wb") as f:
        pickle.dump(store, f)

def runPrompt():
    index = faiss.read_index(index_name)
    with open(namespace, "rb") as f:
        docsearch = pickle.load(f)
    docsearch.index = index

    llm = ChatOpenAI(
        temperature=0, openai_api_key=OPENAI_API_KEY, verbose=True)
    chain = load_qa_chain(llm, chain_type="stuff")

    question_answer_dict = {}
    previous_question = None

    """
    try:

        
        # load config
        load_config()
        # ctrl + c
        sigterm_handler_wrap(signal.SIGINT)
        # kill signal
        sigterm_handler_wrap(signal.SIGTERM)

        # create channel
        channel_name = conf().get("channel_type", "wx")

        if "--cmd" in sys.argv:
            channel_name = "terminal"

        if channel_name == "wxy":
            os.environ["WECHATY_LOG"] = "warn"
            # os.environ['WECHATY_PUPPET_SERVICE_ENDPOINT'] = '127.0.0.1:9001'

        channel = channel_factory.create_channel(channel_name)
        if channel_name in ["wx", "wxy", "terminal", "wechatmp", "wechatmp_service", "wechatcom_app"]:
            PluginManager().load_plugins()

        # startup channel
        channel.startup()

        # Code for book reading
        question_answer_dict = {}
        previous_question = None

        query = input()

        if query.lower() == "repeat":
            if previous_question:
                print("重复上一个问题：", previous_question)
                if previous_question in question_answer_dict:
                    print("回答：", question_answer_dict[previous_question])
                else:
                    print("没有找到上一个问题的回答")
            else:
                print("没有上一个问题")

        elif query.lower() == "context":
            if previous_question:
                print("当前问题基于上一个问题的回答：", previous_question)
                if previous_question in question_answer_dict:
                    context = question_answer_dict[previous_question]
                    docs = docsearch.similarity_search(context, include_metadata=True)
                    r = chain.run(input_documents=docs, question=query)
                    answer = r.encode('utf-8').decode('utf-8')
                    question_answer_dict[query] = answer
                    print("回答：", answer)
                else:
                    print("没有找到上一个问题的回答")
            else:
                print("没有上一个问题")


        else:
            docs = docsearch.similarity_search(query, include_metadata=True)
            r = chain.run(input_documents=docs, question=query)
            answer = r.encode('utf-8').decode('utf-8')
            question_answer_dict[query] = answer
            previous_question = query
            print("回答：", answer)

    except Exception as e:
        logger.error("App startup failed!")
        logger.exception(e)
    
    """

    while True:
        query = input("请问有什么能帮到您的： ")

        if query.lower() == "repeat":
            if previous_question:
                print("重复上一个问题：", previous_question)
                if previous_question in question_answer_dict:
                    print("回答：", question_answer_dict[previous_question])
                else:
                    print("没有找到上一个问题的回答")
            else:
                print("没有上一个问题")

        elif query.lower() == "context":
            if previous_question:
                print("当前问题基于上一个问题的回答：", previous_question)
                if previous_question in question_answer_dict:
                    context = question_answer_dict[previous_question]
                    docs = docsearch.similarity_search(context, include_metadata=True)
                    r = chain.run(input_documents=docs, question=query)
                    answer = r.encode('utf-8').decode('utf-8')
                    question_answer_dict[query] = answer
                    print("回答：", answer)
                else:
                    print("没有找到上一个问题的回答")
            else:
                print("没有上一个问题")
        
        else:
            docs = docsearch.similarity_search(query, include_metadata=True)
            r = chain.run(input_documents=docs, question=query)
            answer = r.encode('utf-8').decode('utf-8')
            question_answer_dict[query] = answer
            previous_question = query
            print("回答：", answer)
        
    #"""
        
        

if __name__ == "__main__":
    if not isExistTrainFile():
        train()
    runPrompt()