import csv
import codecs
import string
import warnings
from scipy import spatial

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn

from sklearn.metrics import f1_score, precision_score, recall_score

class VqaMedEvaluator:
    remove_stopwords = True
    stemming = True
    case_sensitive = False
    def __init__(self, answer_file_path):
        self.answer_file_path = answer_file_path
        self.gt = self.load_gt()
        self.word_pair_dict = {}
    def _evaluate(self, submission_file_path):
        predictions = self.load_predictions(submission_file_path)
        wbss = self.compute_wbss(predictions)
        bleu = self.compute_bleu(predictions)
        accu = self.compute_accu(predictions)
        prec, reca, f1sc = self.compute_prf1(predictions)
        _result_object = {"accuracy": round(accu, 3), "precision": round(prec, 3), "recall": round(reca, 3), "f1": round(f1sc, 3), "bleu": round(bleu, 3),"wbss": round(wbss, 3)}
        return _result_object
    def load_gt(self):
        results = []
        for line in codecs.open(self.answer_file_path,'r','utf-8'):
            QID = line.split('\t')[0]
            ImageID = line.split('\t')[1]
            ans = line.split('\t')[2].strip()
            results.append((QID, ImageID, ans))
        return results
    def load_predictions(self, submission_file_path):
        qa_ids_testset = [tup[0] for tup in self.gt]
        image_ids_testset = [tup[1] for tup in self.gt]
        predictions = []
        occured_qaid_imageid_pairs = []
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            lineCnt = 0
            occured_images = []
            for row in reader:
                lineCnt += 1
                if(len(row) != 3 and len(row) != 2):
                    raise Exception("Wrong format: Each line must consist of an QA-ID followed by a tab, an Image ID, a tab and an answer ({}), where the answer can be empty {}"
                        .format("<QA-ID><TAB><Image-ID><TAB><Answer>", self.line_nbr_string(lineCnt)))
                qa_id = row[0]
                image_id = row[1]
                try:
                    i = qa_ids_testset.index(qa_id)
                    expected_image_id = image_ids_testset[i]
                    if image_id != expected_image_id:
                        raise Exception
                except :
                    raise Exception("QA-ID '{}' with Image-ID '{}' does not represent a valid QA-ID - IMAGE ID pair in the testset {}"
                        .format(qa_id, image_id, self.line_nbr_string(lineCnt)))
                if (qa_id, image_id) in occured_qaid_imageid_pairs:
                    raise Exception("The QA-ID '{}' with Image-ID '{}' pair appeared more than once in the submission file {}"
                        .format(qa_id, image_id, self.line_nbr_string(lineCnt)))
                answer = row[2] if (len(row) == 3) else ""
                predictions.append((qa_id, image_id, answer))
                occured_qaid_imageid_pairs.append((qa_id, image_id))
            if len(predictions) != len(self.gt):
                raise Exception("Number of QA-ID - Image-ID pairs in submission file does not correspond with number of QA-ID - Image-ID pairs in testset")
        return predictions
    def compute_wbss(self, predictions):
        nltk.download('wordnet')
        count = 0
        totalscore_wbss = 0.0
        for tuple1, tuple2 in zip(self.gt, predictions):
            QID1 = tuple1[0]
            QID2 = tuple2[0]
            imageID1 = tuple1[1]
            imageID2 = tuple2[1]
            ans1 = tuple1[2]
            ans2 = tuple2[2]
            assert (QID1 == QID2)
            assert (imageID1 == imageID2)
            count+=1
            QID = QID1
            if ans1==ans2:
                score_wbss = 1.0
            elif ans2.strip() == "":
                score_wbss = 0
            else:
                score_wbss = self.calculateWBSS(ans1,ans2)
            totalscore_wbss+=score_wbss
        return totalscore_wbss/float(count)
    def compute_accu(self, predictions):
        count = 0
        totalscore_accu = 0.0
        for tuple1, tuple2 in zip(self.gt, predictions):
            QID1 = tuple1[0]
            QID2 = tuple2[0]
            imageID1 = tuple1[1]
            imageID2 = tuple2[1]
            ans1 = tuple1[2]
            ans2 = tuple2[2]
            assert (QID1 == QID2)
            assert (imageID1 == imageID2)
            count+=1
            QID = QID1
            if ans1==ans2:
                score_accu = 1.0
            else:
                score_accu = 0.0
            totalscore_accu+=score_accu
        return totalscore_accu/float(count)
    def compute_prf1(self, predictions):
        true,pred = [],[]
        totalscore_accu = 0.0
        for tuple1, tuple2 in zip(self.gt, predictions):
            QID1 = tuple1[0]
            QID2 = tuple2[0]
            imageID1 = tuple1[1]
            imageID2 = tuple2[1]
            ans1 = tuple1[2]
            ans2 = tuple2[2]
            assert (QID1 == QID2)
            assert (imageID1 == imageID2)
            QID = QID1
            true.append(ans1)
            pred.append(ans2)
            p = precision_score(true, pred, average='macro')
            r = recall_score(true, pred, average='macro')
            f1 = f1_score( true, pred, average='macro' )
        return p, r, f1
    def calculateWBSS(self,S1, S2):
        if S1 is None or S2 is None:
            return 0.0
        dictionary = self.constructDict(S1.split(), S2.split())
        vector1 = self.getVector_wordnet(S1, dictionary)
        vector2 = self.getVector_wordnet(S2, dictionary)
        cos_similarity = self.calculateCosineSimilarity(vector1, vector2)
        return cos_similarity
    def getVector_wordnet(self,S, dictionary):
        vector = [0.0]*len(dictionary)
        for index, word in enumerate(dictionary):
            for wordinS in S.split():
                if wordinS == word:
                    score = 1.0
                else:
                    score = self.wups_score(word,wordinS)
                if score > vector[index]:
                    vector[index]=score
        return vector
    def constructDict(self, list1, list2):
        return list(set(list1+list2))
    def wups_score(self,word1, word2):
        score = 0.0
        score = self.wup_measure(word1, word2)
        return score
    def wup_measure(self,a, b, similarity_threshold = 0.925, debug = False):
        if debug: print('Original', a, b)
        if a+','+b in self.word_pair_dict.keys():
            return  self.word_pair_dict[a+','+b]
        def get_semantic_field(a):
            return wn.synsets(a, pos=wn.NOUN)
        if a == b: return 1.0
        interp_a = get_semantic_field(a)
        interp_b = get_semantic_field(b)
        if debug: print(interp_a)
        if interp_a == [] or interp_b == []:
            return 0.0
        if debug: print('Stem', a, b)
        global_max=0.0
        for x in interp_a:
            for y in interp_b:
                local_score=x.wup_similarity(y)
                if debug: print('Local', local_score)
                if local_score > global_max:
                    global_max=local_score
        if debug: print('Global', global_max)
        if global_max < similarity_threshold:
            interp_weight = 0.1
        else:
            interp_weight = 1.0
        final_score = global_max * interp_weight
        self.word_pair_dict[a+','+b] = final_score
        return final_score
    def calculateCosineSimilarity(self, vector1, vector2):
        return 1-spatial.distance.cosine(vector1, vector2)
    def compute_bleu(self, predictions):
        warnings.filterwarnings('ignore')
        nltk.download('punkt')
        nltk.download('stopwords')
        stops = set(stopwords.words("english"))
        stemmer = SnowballStemmer("english")
        translator = str.maketrans('', '', string.punctuation)
        candidate_pairs = self.readresult(predictions)
        gt_pairs = self.readresult(self.gt)
        max_score = len(gt_pairs)
        current_score = 0
        i = 0
        for image_key in candidate_pairs:
            candidate_caption = candidate_pairs[image_key]
            gt_caption = gt_pairs[image_key]
            if not VqaMedEvaluator.case_sensitive:
                candidate_caption = candidate_caption.lower()
                gt_caption = gt_caption.lower()
            candidate_words = nltk.tokenize.word_tokenize(candidate_caption.translate(translator))
            gt_words = nltk.tokenize.word_tokenize(gt_caption.translate(translator))
            if VqaMedEvaluator.remove_stopwords:
                candidate_words = [word for word in candidate_words if word.lower() not in stops]
                gt_words = [word for word in gt_words if word.lower() not in stops]
            if VqaMedEvaluator.stemming:
                candidate_words = [stemmer.stem(word) for word in candidate_words]
                gt_words = [stemmer.stem(word) for word in gt_words]
            try:
                if len(gt_words) == 0 and len(candidate_words) == 0:
                    bleu_score = 1
                else:
                    bleu_score = nltk.translate.bleu_score.sentence_bleu([gt_words], candidate_words, smoothing_function=SmoothingFunction().method0)
            except ZeroDivisionError:
                pass
            current_score += bleu_score
        return current_score / max_score
    def readresult(self,tuples):
        pairs = {}
        for row in tuples:
             pairs[row[0]]=row[2]
        return pairs
    def line_nbr_string(self, line_nbr):
        return "(Line nbr {})".format(line_nbr)


