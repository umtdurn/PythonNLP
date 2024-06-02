# uvicorn api-main:app --reload   --> bash code to run auto-reloadable apis
from fastapi import FastAPI
from main import *

app = FastAPI()

@app.get("/fasttext_recommendations/")
def get_fasttext_recommendations(keywords: str,mail: str):
    # Anahtar kelimeleri bir listeye dönüştür
    keyword_list = keywords.split(',')
    # Bu anahtar kelimelere göre makaleleri bulup döndüreceğiz
    user_fasttext_vector = get_user_fasttext_vector(keyword_list,mail)
    article_fasttext_vector = get_article_fasttext_vector_dict()
    recommended_articles = get_recommendations(user_fasttext_vector,article_fasttext_vector)
    return recommended_articles

@app.get("/scibert_recommendations/")
def get_scibert_recommendations(keywords: str, mail: str):
    keyword_list = keywords.split(',')
    user_scibert_vector = get_user_scibert_vector(keyword_list,mail)
    article_scibert_vector = get_article_scibert_vector_dict()
    recommended_articles = get_recommendations(user_scibert_vector[0],article_scibert_vector)
    return recommended_articles




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
