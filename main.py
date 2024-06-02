from datasets import load_dataset
import spacy
from transformers import AutoTokenizer, AutoModel
import fasttext.util
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import numpy as np

# dataset = load_dataset("memray/krapivin")

client = MongoClient('mongodb://localhost:27017')
db = client['ArticleRecommendation']

article_collection = db['ArticleDataset']
article_vectors_collection = db['ArticleVectors']
user_collection = db['Users']
liked_articles_collection = db['LikedArticles']
dissliked_articles_collection = db['DisslikedArticles']

# validation_data_for_store = []
# test_data_for_store = []

# for row in dataset['validation']:
#     insert_object = {
#         "name": row['name'],
#         "title": row['title'],
#         "abstract": row['abstract'],
#         "fulltext": row['fulltext'],
#         "keywords": row['keywords'],
#     }
#     validation_data_for_store.append(insert_object)

# for row in dataset['test']:
#     insert_object = {
#         "name": row['name'],
#         "title": row['title'],
#         "abstract": row['abstract'],
#         "fulltext": row['fulltext'],
#         "keywords": row['keywords'],
#     }
#     test_data_for_store.append(insert_object)

# validation_data = {
#     "name": "validation",
#     "data": validation_data_for_store
# }
# test_data = {
#     "name": "test",
#     "data": test_data_for_store
# }

# for i in validation_data_for_store:
#     article_collection.insert_one({
#         "name": "validation",
#         "data": i
#     })

# for i in test_data_for_store:
#     article_collection.insert_one({
#         "name": "test",
#         "data": i
#     })


# SCIBERT modeli için gerekli işlemler
# print(1)
# tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased',force_download=True)
# model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased',force_download=True)
# print(2)
# Huggingface'deki veri setini yükle

# Kullanıcı listesi oluşturuluyor
# users = [
#     {
#         "first_name": "Alice",
#         "last_name": "Johnson",
#         "birth_date": "1990-05-15",
#         "gender": "Female",
#         "email": "alice.johnson@example.com",
#         "interests": ["artificial intelligence", "machine learning", "data mining"]
#     },
#     {
#         "first_name": "Bob",
#         "last_name": "Smith",
#         "birth_date": "1985-08-25",
#         "gender": "Male",
#         "email": "bob.smith@example.com",
#         "interests": ["computer graphics", "virtual reality"]
#     },
#     {
#         "first_name": "Carol",
#         "last_name": "White",
#         "birth_date": "1992-11-30",
#         "gender": "Female",
#         "email": "carol.white@example.com",
#         "interests": ["neuroscience", "cognitive science"]
#     }
# ]
users = [
    {"first_name": "Alice", "last_name": "Johnson", "interests": ["artificial intelligence", "machine learning", "data mining"]}
]

# print(3)
# spaCy'nin İngilizce dil modelini yükle
nlp = spacy.load("en_core_web_sm")

# spaCy'nin İngilizce dil modelini yükle
def preprocess_text(text):
    # Metni işle
    doc = nlp(text)
    # Stopwords'ü çıkar, lemmatize et ve noktalama işaretlerini çıkar
    cleaned_text = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    # Listeyi string'e çevir
    cleaned_text = " ".join(cleaned_text)
    return cleaned_text
# print(4)
# FastText vektör temsili metotu
fasttext.util.download_model('en', if_exists='ignore')  # İngilizce modeli indir
ft_model = fasttext.load_model('cc.en.300.bin')
# print(5)
def get_fasttext_vector(text):
    return ft_model.get_sentence_vector(text)

def get_average_fasttext_vector(texts):
    vectors = [ft_model.get_sentence_vector(text) for text in texts]
    average_vector = np.mean(vectors, axis=0)
    return average_vector

def get_average_scibert_vector(texts):
    vectors = [get_scibert_vector(text) for text in texts]
    average_vector = np.mean(vectors, axis=0)
    return average_vector

# SCIBERT vektör temsili metotu
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
# print(6)
def get_scibert_vector(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = scibert_model(**inputs)
    # Çıktının son katmanının ortalamasını al
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# def adjust_interest_vector(user_interest_vector, liked_article_vectors, disliked_article_vectors):
    # Kullanıcı ilgi alanları vektörünü başlangıç olarak ayarla
    adjustments = user_interest_vector.copy()
    total_vectors = 1  # Başlangıçta sadece kullanıcı ilgi alanları vektörü var
    # Eğer beğenilen makaleler varsa, vektörlerini topla
    liked_article_vectors = np.array(liked_article_vectors)
    disliked_article_vectors = np.array(disliked_article_vectors)
    if len(liked_article_vectors) > 0:
        positive_adjustments = np.sum(liked_article_vectors, axis=0)
        adjustments += positive_adjustments
        total_vectors += len(liked_article_vectors)  # Beğenilen makale sayısını toplam vektöre ekle
    # Eğer beğenilmeyen makaleler varsa, vektörlerini çıkar
    if len(disliked_article_vectors) > 0:
        negative_adjustments = np.sum(disliked_article_vectors, axis=0)
        adjustments -= negative_adjustments
        total_vectors += len(disliked_article_vectors)  # Beğenilmeyen makale sayısını toplam vektöre ekle
    # Toplam vektör sayısına bölerek ortalamayı hesapla
    adjusted_vector = adjustments / total_vectors
    return adjusted_vector


# def adjust_interest_vector(user_interest_vector, liked_article_vectors, disliked_article_vectors):
    adjustments = user_interest_vector.copy()
    # Beğenilen makale vektörlerini topla
    if len(liked_article_vectors) > 0:
        positive_adjustments = np.sum(liked_article_vectors, axis=0)
        adjustments += positive_adjustments
    # Beğenilmeyen makale vektörlerini çıkar
    if len(disliked_article_vectors) > 0:
        negative_adjustments = np.sum(disliked_article_vectors, axis=0)
        adjustments -= negative_adjustments
    # Normalizasyon yap
    if np.linalg.norm(adjustments) != 0:
        adjusted_vector = adjustments / np.linalg.norm(adjustments)
    else:
        adjusted_vector = adjustments  # Eğer norm 0 ise, değişiklik yapılmamış demektir.
    return adjusted_vector

def adjust_interest_vector(user_interest_vector, liked_article_vectors, disliked_article_vectors):
    # Beğenilen makale vektörlerinin ortalamasını hesapla
    if len(liked_article_vectors) > 0:
        print("liked if'in içerisine girdi..")
        liked_mean_vector = np.mean(liked_article_vectors, axis=0)
        # Kullanıcının ilgi alanları vektörüne beğenilen makalelerin ortalamasını ekle
        user_interest_vector += liked_mean_vector

    # Beğenilmeyen makale vektörlerinin ortalamasını hesapla
    if len(disliked_article_vectors) > 0:
        print("disliked if'in içerisine girdi..")
        disliked_mean_vector = np.mean(disliked_article_vectors, axis=0)
        # Kullanıcının ilgi alanları vektöründen beğenilmeyen makalelerin ortalamasını çıkar
        user_interest_vector -= disliked_mean_vector

    return user_interest_vector

# Cosine similarity hesaplama
def get_recommendations(user_vector, article_vectors):
    similarities = {title: cosine_similarity([user_vector], [vec])[0][0] for title, vec in article_vectors.items()}
    # Benzerlik skorlarına göre sırala ve en yüksek skorlu 5 makaleyi getir
    recommended_articles = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:5]
    return recommended_articles

# def get_user_fasttext_vector(interest_list):
#     return get_average_fasttext_vector(interest_list)

def get_user_fasttext_vector(interest_list, user_mail):
    # user_mail = "210202004@kocaeli.edu.tr"
    user_interest_fasttext_vector = get_average_fasttext_vector(interest_list)
    query_result_like = liked_articles_collection.find({"UserEmail": user_mail})
    query_result_disslike = dissliked_articles_collection.find({"UserEmail": user_mail})

    print(type(user_mail))

    user_liked_article_list = []
    for document in query_result_like:
        user_liked_article_list.append(document["Title"])
        print(document["Title"])
        
        
    user_dissliked_article_list = []
    for document in query_result_disslike:
        user_dissliked_article_list.append(document["Title"])
        print(document["Title"])


    liked_articles_vectors_test = []
    if(len(user_liked_article_list) > 0 ):
        query_result2 =  article_vectors_collection.find({"title": {"$in": user_liked_article_list}})
        for document in query_result2:
            liked_articles_vectors_test.append(document["fast_text_vector"])


    dissliked_articles_vectors_test = []
    if(len(user_dissliked_article_list) > 0):
        query_result3 = article_vectors_collection.find({"title": {"$in": user_dissliked_article_list}})
        for document in query_result3:
            dissliked_articles_vectors_test.append(document["fast_text_vector"])

    user_adjusted_vector = adjust_interest_vector(user_interest_fasttext_vector,liked_articles_vectors_test,dissliked_articles_vectors_test)
    return user_adjusted_vector

# def get_user_scibert_vector(interest_list):
#     return get_average_scibert_vector(interest_list)

def get_user_scibert_vector(interest_list,user_mail):
    user_interest_scibert_vector = get_average_scibert_vector(interest_list)
    query_result_like = liked_articles_collection.find({"UserEmail": user_mail})
    query_result_disslike = dissliked_articles_collection.find({"UserEmail": user_mail})


    user_liked_article_list = []
    for document in query_result_like:
        user_liked_article_list.append(document["Title"])
        print(document["Title"])
        
        
    user_dissliked_article_list = []
    for document in query_result_disslike:
        user_dissliked_article_list.append(document["Title"])
        print(document["Title"])

    liked_articles_vectors_test = []
    if(len(user_liked_article_list) > 0 ):
        query_result2 =  article_vectors_collection.find({"title": {"$in": user_liked_article_list}})
        for document in query_result2:
            liked_articles_vectors_test.append(document["scibert_vector"][0])
            print(document["scibert_vector"][0])


    dissliked_articles_vectors_test = []
    if(len(user_dissliked_article_list) > 0):
        query_result3 = article_vectors_collection.find({"title": {"$in": user_dissliked_article_list}})
        for document in query_result3:
            dissliked_articles_vectors_test.append(document["scibert_vector"][0])
            print(document["scibert_vector"][0])

    user_adjusted_vector = adjust_interest_vector(user_interest_scibert_vector,liked_articles_vectors_test,dissliked_articles_vectors_test)
    return user_adjusted_vector


def get_article_fasttext_vector_dict():
    article_fasttext_vectors_dict = {}
    article_documents = article_vectors_collection.find({"type": "validation"})
    for document in article_documents:
        if "fast_text_vector" in document and "title" in document:
            article_fasttext_vectors_dict[document["title"]] = document["fast_text_vector"]
    return article_fasttext_vectors_dict

def get_article_scibert_vector_dict():
    article_scibert_vectors_dict = {}
    article_documents = article_vectors_collection.find({"type": "validation"})
    for document in article_documents:
        if "scibert_vector" in document and "title" in document:
            article_scibert_vectors_dict[document["title"]] = document["scibert_vector"][0]
    return article_scibert_vectors_dict



# Dataset'ten ilk makalenin özetini al ve ön işleme uygula
# sample_abstract = dataset['validation'][0]['abstract']
# processed_text = preprocess_text(sample_abstract)
# print("Orijinal Metin:", sample_abstract)
# print("İşlenmiş Metin:", processed_text)

# for user in users:
#     interests_text = ' '.join(user['interests'])
#     fasttext_vector = get_fasttext_vector(interests_text)
#     scibert_vector = get_scibert_vector(interests_text)
#     print(f'User: {user["first_name"]} {user["last_name"]}')
#     print('FastText Vector:', fasttext_vector)
#     print('SCIBERT Vector:', scibert_vector)

# article_vectors_fasttext = {article['title']: get_fasttext_vector(article['abstract']) for article in dataset['validation']}
# article_vectors_scibert = {article['title']: get_scibert_vector(article['abstract']) for article in dataset['validation']}

interests_text = ' '.join(users[0]['interests'])
user_vector_fasttext = get_fasttext_vector(interests_text)
user_vector_scibert = get_scibert_vector(interests_text)

# recommendsByFastText = get_recommendations(user_vector_fasttext, article_vectors_fasttext)
# recommendsBySciBert = get_recommendations(user_vector_scibert, article_vectors_scibert)
# ------------------------------------------------------------------------------------------------------- #
# article_vectors_fasttext = {}
# article_vectors_scibert = {}

# for article in dataset['validation']:
#     try:
#         title = article['title']
#         abstract = article['abstract']
#         article_vectors_fasttext[title] = get_fasttext_vector(abstract)
#         article_vectors_scibert[title] = get_scibert_vector(abstract)
        
#         insert_data = {
#             "type": "validation",
#             "title": title,
#             "fast_text_vector": article_vectors_fasttext[title].tolist(),
#             "scibert_vector": article_vectors_scibert[title].tolist()
#         }
#         article_vectors_collection.insert_one(insert_data)
#         # print(article_vectors_fasttext[title])
#         # print(article_vectors_scibert[title])
#     except TypeError as e:
#         print(f"Error processing article: {article}")
#         print(f"TypeError: {e}")

# print('Makale vektör temsillerini çıkartma işlemi bitti.')
# ------------------------------------------------------------------------------------------------------- #

# interests_text = ' '.join(users[0]['interests'])
# # interests_text = interests_text + 'remapping ' + 'mapping ' + 'refinement ' + 'parallel ' + 'linear-programming '
# user_vector_fasttext = get_fasttext_vector(interests_text)
# user_vector_scibert = get_scibert_vector(interests_text)

# user_vector_as_list = user_vector_fasttext.tolist()

# print(type(user_vector_fasttext))
# print(user_vector_fasttext)
# print(user_vector_fasttext.shape)

# print(type(user_vector_as_list))
# print(user_vector_as_list)

# print('Kullanıcı vektör temsillerini çıkartma işlemi bitti.')

# recommendsByFastText = get_recommendations(user_vector_fasttext, article_vectors_fasttext)
# print(user_vector_fasttext.shape)
# print(article_vectors_fasttext.shape)
# recommendsBySciBert = get_recommendations(user_vector_scibert, article_vectors_scibert)

# print('Önerileri alma işlemleri bitti..')

# print(recommendsByFastText)
# print(recommendsBySciBert)
# print('---------------------------------')
# article_vectors_fasttext = {article['title']: get_fasttext_vector(article['abstract']) for article in dataset['validation']}

# user_from_db = user_collection.find({"FirstName": "Alice"})
# for i in user_from_db:
#     interests = i['Interests']

# average_scibert_vector = get_average_scibert_vector(interests)
# article_vectors_fasttext = {}
# article_vectors_scibert = {}
# article_documents = article_vectors_collection.find({"type": "validation"})
# # # 'fast_text_vector' ve 'title' değerlerini içeren obje listesini oluşturma

# for document in article_documents:
#     if "fast_text_vector" in document and "title" in document:
#         article_vectors_fasttext[document["title"]] = document["fast_text_vector"]

# article_documents2 = article_vectors_collection.find({"type": "validation"})
# for document in article_documents2:
#     if "scibert_vector" in document and "title" in document:
#         article_vectors_scibert[document["title"]] = document["scibert_vector"][0]
#         # print(document['title'])

# tmp1 = article_vectors_fasttext["Closure properties of constraints."]
# tmp2 = article_vectors_scibert["Closure properties of constraints."]
# # print(np.array(tmp1).shape)
# # print(np.array(tmp2).shape)
# tmp3 = np.array(tmp2).shape

# # fast_text_recomm = get_recommendations(user_vector_fasttext,article_vectors_fasttext)
# # print(fast_text_recomm)
# print(np.array(user_vector_fasttext).shape)
# print(np.array(user_vector_fasttext).shape)
# print('----------------------------------------')
# print(np.array(user_vector_scibert[0]).shape)
# print(np.array(tmp2).shape)

# scibert_recomm = get_recommendations(user_vector_scibert[0],article_vectors_scibert)
# print(scibert_recomm)


# print('----------------------------------------')

# interest_list2 = [".Net Development","Machine Learning"]
# user_interest_fasttext_vector = get_average_fasttext_vector(interest_list2)
# query_result_like = liked_articles_collection.find({"UserEmail": "210202004@kocaeli.edu.tr"})
# query_result_disslike = dissliked_articles_collection.find({"UserEmail": "210202004@kocaeli.edu.tr"})

# user_liked_article_list = []
# for document in query_result_like:
#     user_liked_article_list.append(document["Title"])
    
# user_dissliked_article_list = []
# for document in query_result_disslike:
#     user_dissliked_article_list.append(document["Title"])

# liked_articles_vectors_test = []
# if(len(user_liked_article_list) > 0 ):
#     query_result2 =  article_vectors_collection.find({"title": {"$in": user_liked_article_list}})
#     for document in query_result2:
#         liked_articles_vectors_test.append(document["fast_text_vector"])

# dissliked_articles_vectors_test = []
# if(len(user_dissliked_article_list) > 0):
#     query_result3 = article_vectors_collection.find({"title": {"$in": user_dissliked_article_list}})
#     for document in query_result3:
#         dissliked_articles_vectors_test.append(document["fast_text_vector"])

# print('LİKEEEEEEEEED')
# print(np.array(liked_articles_vectors_test).shape)

# print('DİSLİKEEEEEEEED')
# print(np.array(dissliked_articles_vectors_test).shape)

# tmp1 = np.array(article_vectors_fasttext['Practical Algorithms for Selection on Coarse-Grained Parallel Computers'])
# tmp2 = np.array(article_vectors_scibert['Practical Algorithms for Selection on Coarse-Grained Parallel Computers'])
# tmp3 = np.array(article_vectors_scibert.get(0)[0])
# print(tmp1.shape)
# print(tmp2.shape)
# print(type(tmp3))



# tmp_vector = vector_list[0]['scibert_vector'][0]
# print(np.array(tmp_vector).shape)
# print(user_vector_fasttext.shape)


# numpy_array = np.array(vector_list)
# recomm = get_recommendations(user_vector_fasttext,article_vectors_fasttext2)
# print(recomm)


# print(numpy_array.shape)
# print('---------------------------------')
# numpy_array = numpy_array.reshape(1, -1)
# print(numpy_array.shape)