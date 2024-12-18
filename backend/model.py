import numpy as np
import os
from dataset import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
import pickle

class Model():
    def __init__(self, corpus: Dataset, query_path = None, load_pickle = False, **kargs) -> None:
        if not isinstance(corpus, Dataset):
            raise TypeError("Corpus must be in Dataset type!")
        
        if query_path:
            if not os.path.isfile(query_path):
                raise FileNotFoundError("Query file does not exist!")
            else:
                query = Dataset(punct_file=corpus.get_punct_file(), stopword_file=corpus.get_stopword_file(), data_path=query_path)
        
        else:
            for _, v in kargs.items():
                query = Dataset(punct_file=corpus.get_punct_file(), stopword_file=corpus.get_stopword_file(), data = v)

        self._corpus = corpus.get_raw_data()
        self._query = query.get_raw_data()

        self._model = TfidfVectorizer(min_df=2, max_df=0.8, sublinear_tf=True)

        self._corpus_embeddings = self._model.fit_transform(corpus.get_normalized_data()).toarray()
        self._query_embeddings = self._model.transform(query.get_normalized_data()).toarray()

        self._sim_matrix = cosine_similarity(self._query_embeddings, self._corpus_embeddings)
    
    
    def compute_similarity_maxtrix(self):
        return self._sim_matrix
    
    def get_top_k(self, k: int):
        if not isinstance(k, int):
            raise TypeError("K must be integer value!")

        if k > len(self._corpus_embeddings):
            raise ValueError("K must be smaller or equal to corpus size!")
        
        results = []
        for idx in range(self._sim_matrix.shape[0]):
            indices = np.argsort(self._sim_matrix[idx])[:-k-1:-1]
            tmp = []
            for item in indices:
                tmp.append({
                    'sim_value': self._sim_matrix[idx][item],
                    'indices': item,
                    'corpus': self._corpus[item],
                })
            results.append({
                'id': idx,
                'query': self._query[idx],
                # 'sim_value': self._sim_matrix[idx][indices].tolist(),
                # 'indices': indices.tolist(),
                # 'corpus': np.array(self._corpus)[indices].tolist(),
                'result': tmp
            })
        # pprint(*results)
        return results


if __name__ == "__main__":
    corpus = Dataset(punct_file="./punctuation.txt", stopword_file="./stopwords.txt", data_path="./applications.json")
    query = ["<p>Louis Gallant đứa trẻ mồi côi sống tại khu ổ chuột phía Nam Mexico, nơi đây vốn được xem là 1 trong những thành phố nguy hiểm nhất thế giới. Việc các băng đảng thanh toán lẫn nhau giao dịch cần sa, ma túy và mại dâm xảy ra thằng ngày nên nơi đây coi như là thiên đường của tội phạm. Từ nhỏ Luis đã sống trong cái nôi của tội ác, đi theo các đứa bạn cùng trang lứa để trộm cắp lừa gặt để lấy tiền tiêu xài. Để thay đổi cuộc đời mình Luis đã tìm đến ông trùm ma túy lớn nhất vùng muốn trở thành đối tác của ông ta, Ông trùm nhìn Luis chỉ là thằng nhóc miệng còn nhôi sữa nên không chấp dứt. Nhưng Luis một mựt muốn hợp tác, Ông Trùm thấy vậy lấy ra một chiếc áo chống đạn để Luis chứng minh bản thân mình, Luis không thề do dự mặc chiếc áo vào và để để ông trùm chỉa súng vào mình. Có lẻ Ông Trùm thấy hình bóng của mình hồi trẻ trên người Luis nên cố tình bắn lệt cho Luis một cơ hội, cũng vì vậy Luis đổi lại được 1kg ma túy. Khi về Luis bắt đầu đi tìm những kẻ nghiện ngập trong các con hẻm trong khu phố để bán, nhìn những sấp tiền trên tay mình khóe miệng bất giác cong lên Luis chưa bao giờ cảm thấy kiếm tiền dễ dàng như thế. Sau đó trong đầu Luis xuất hiện 1 ý nghĩ điên cuồng Luis lại đến tìm Ông trùm trước đó, khi đến lúc đó ông trùm đang gặp phiền toái, 1 tên đàn em của ông trùm khi đi giao dịch đã bị cướp hàng và dính một viên đạn, muốn cứu tên đàn em phải đưa đến bệnh viện nhưng tài xế cũng phải đối mặt nguy cơ ngồi tù nếu như lộ diện. Luis lúc này nảy số nếu cho tôi theo ông tôi sẽ có cách đưa người anh em này đến bệnh viện, Luis đưa người anh em đó đến bệnh viện và bỏ mặt sống chết ở đó không quan tâm. Nhờ sự liều lĩnh và táo bạo đó ôm trùm rất thích và cho Luis làm đàn em của mình, nhờ sự trung thành và gang dạ của mình Luis đã là 1 trong những tên đàn em được ông tin tưởng nhất. Trong một lần ông trùm đi du lịch đã bị sát hại, Evan một thằng đàn em của ông trùm đã liên hệ để điều tra và trả thù. Sau một tháng điều tra chúng tôi biết được người sát hại là một băng nhóm mới xuất hiện cùng khu phố. Chúng tôi lấy danh nghĩ của một doanh nhân hẹn gặp tại một khu cảng sát biên giới giữa Mỹ và Mexico để giao dịch. Luis và Evan cùng 1 số anh em setup tại bến cảng nhằm mục đích trả thù và cướp hàng, lúc giao chiến vì thực lực bên kia quá mạnh nên chúng tôi đành rút lui, Evan bị dính đạn vào chân biết không thoát được nên đã liều mình ở lại cầm chân để cho Luis chạy thoát. Luis nhanh chân chạy lên 1 con tàu trốn trong 1 thùng container Luis ngất đi vì quá mệt, khi tỉnh dậy Luis thấy mình đang ở trong một khu cảng khác. Luis đi tìm những ngư dân xung quanh để hỏi thăm thì té ngữa ra là vì biết mình đang ở Los Santos, Luis được 1 người dân giới thiệu đế phía Nam Southside khu vực nơi người dân Mexico tụ hợp. Đến đây Luis quen một người chị tên Ell LeFant, Luis kể hoàng cảnh của mình cho người chị nghe và được người chị thông cảm và nhờ mối quan hệ của mình làm cho Luis một Passport để nhập cứ chính ngạch và sinh sống tại Los Santos....</p>"]
    obj = Model(corpus=corpus, query=query)
    obj.get_top_k(3)