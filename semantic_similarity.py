import pandas as pd
import numpy as np
## Scikit-Learn Utilities:
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import HTML

## AdaptNLP Utilities:
from adaptnlp import EasyWordEmbeddings, EasyStackedEmbeddings, EasyDocumentEmbeddings


## Load biobert model & pre-trained embeddings:
biobert_embeddings = EasyDocumentEmbeddings("biobert/biobert_v1.1_pubmed_pytorch_model/")
biobert_ls = np.load('biobert/biobert_pre_trained_embeddings/biobert_covid_19_embeddings.npy')

def get_similar_articles(search_string, title_ls, results_returned):
	sent_emb = biobert_embeddings.embed_pool(search_string)
	search_vect = [i.get_embedding() for i in sent_emb][0]
	search_vect = search_vect.cpu().detach().numpy()
	cosine_similarities = pd.Series(cosine_similarity([search_vect], biobert_ls).flatten())
	
	output = []
	for i,j in cosine_similarities.nlargest(int(results_returned)).iteritems():
		output.append(title_ls[i])

	results = "\n".join(output)
	return results