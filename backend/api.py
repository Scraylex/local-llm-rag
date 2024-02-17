from flask import Flask, request

from retrieval_qa_client import RetrievalQAClient

app = Flask(__name__)
rag_client = RetrievalQAClient()


@app.route('/query', methods=['POST'])
def query_endpoint():
    # Get the query parameter from the request
    user_query = request.json.get('question')
    result = rag_client.do_rag(user_query)
    return f"Query result: {result['result']}"


if __name__ == '__main__':
    app.run()
