# %%
import torch
import flask
from flask import Flask, request, render_template
import json
from rank_models import tfidf, bm25_model
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')

## Semantic Similarity:
from semantic_similarity import get_similar_articles

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_answer', methods=['POST'])
def get_answer():
    try:
        query = request.json['input_question']
        num_paragraphs = int(request.json['num_paragraphs'])
        query = query.lstrip().rstrip()
        text_ls = list(np.load('data/title_list.npy'))
        if len(text_ls) > 0:
            bm_1, _, _ = bm25_model.get_similarity([query], text_ls)
            bm_1 = np.array(bm_1)
            bm_1_idx = bm_1[bm_1[:, 1] > 1][:num_paragraphs, 0]  # two most similar
            bm_1_idx = np.array(bm_1_idx, dtype=int)
            text = '\n'.join(text_ls[i] for i in sorted(bm_1_idx))
            print('======= BM25 SCORES =======')
            print(bm_1)
            if len(bm_1_idx) == 0:
                return app.response_class(response=json.dumps("Text passages not found. Provide more information in your query"), status=500, mimetype='application/json')

            # Generate response
            res_biobert = get_similar_articles(query, text_ls, num_paragraphs)

            res =  {'biobert': res_biobert,
                   'text_paragraphs': text}
            
            return flask.jsonify(res)
        else:
            return app.response_class(response=json.dumps("Provide more information in your query"), status=500, mimetype='application/json')
    except Exception as error:
        res = str(error)
        return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000, use_reloader=False)
