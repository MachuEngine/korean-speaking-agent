import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# 토픽별 한국어 표현 샘플 데이터
KOREAN_PHRASES: list[Document] = [
    # ── 여행 (Travel) ──────────────────────────────────────────────
    Document(
        page_content=(
            "공항 표현: 탑승구가 어디예요? (Where is the gate?) / "
            "수하물을 찾는 곳이 어디예요? (Where is baggage claim?) / "
            "비행기 표를 예매했어요 (I booked a flight ticket)"
        ),
        metadata={"topic": "여행", "subtopic": "공항"},
    ),
    Document(
        page_content=(
            "호텔 표현: 체크인하고 싶어요 (I'd like to check in) / "
            "방을 예약했어요 (I reserved a room) / "
            "짐을 맡길 수 있어요? (Can I leave my luggage?)"
        ),
        metadata={"topic": "여행", "subtopic": "호텔"},
    ),
    Document(
        page_content=(
            "관광 표현: 이 근처에 관광 명소가 있어요? (Are there tourist spots nearby?) / "
            "사진 찍어도 돼요? (May I take a photo?) / "
            "입장료가 얼마예요? (How much is the admission fee?)"
        ),
        metadata={"topic": "여행", "subtopic": "관광"},
    ),
    Document(
        page_content=(
            "길 묻기: 지도를 보여줄 수 있어요? (Can you show me a map?) / "
            "여기서 얼마나 걸려요? (How long does it take from here?) / "
            "길을 잃었어요 (I'm lost)"
        ),
        metadata={"topic": "여행", "subtopic": "길찾기"},
    ),
    # ── 일상 (Daily Life) ──────────────────────────────────────────
    Document(
        page_content=(
            "인사 표현: 좋은 아침이에요 (Good morning) / "
            "잘 잤어요? (Did you sleep well?) / "
            "오늘 뭐 해요? (What are you doing today?)"
        ),
        metadata={"topic": "일상", "subtopic": "인사"},
    ),
    Document(
        page_content=(
            "쇼핑 표현: 이거 얼마예요? (How much is this?) / "
            "다른 색깔은 없어요? (Are there other colors?) / "
            "좀 깎아줄 수 있어요? (Can you give me a discount?)"
        ),
        metadata={"topic": "일상", "subtopic": "쇼핑"},
    ),
    Document(
        page_content=(
            "교통 표현: 지하철역이 어디예요? (Where is the subway station?) / "
            "버스가 몇 번이에요? (What bus number is it?) / "
            "택시 좀 불러주세요 (Please call a taxi)"
        ),
        metadata={"topic": "일상", "subtopic": "교통"},
    ),
    # ── 취미 (Hobbies) ─────────────────────────────────────────────
    Document(
        page_content=(
            "운동 표현: 운동을 좋아해요? (Do you like exercising?) / "
            "헬스장에 자주 가요 (I often go to the gym) / "
            "어떤 운동을 해요? (What kind of exercise do you do?)"
        ),
        metadata={"topic": "취미", "subtopic": "운동"},
    ),
    Document(
        page_content=(
            "독서 표현: 책 읽는 것을 좋아해요 (I like reading books) / "
            "요즘 무슨 책을 읽고 있어요? (What book are you reading lately?) / "
            "추천해 줄 책이 있어요? (Do you have a book to recommend?)"
        ),
        metadata={"topic": "취미", "subtopic": "독서"},
    ),
    Document(
        page_content=(
            "음악 표현: 어떤 음악을 좋아해요? (What kind of music do you like?) / "
            "악기를 연주할 수 있어요? (Can you play an instrument?) / "
            "콘서트에 가 본 적 있어요? (Have you ever been to a concert?)"
        ),
        metadata={"topic": "취미", "subtopic": "음악"},
    ),
    # ── 음식 (Food) ────────────────────────────────────────────────
    Document(
        page_content=(
            "식당 표현: 메뉴판 좀 주세요 (Please give me the menu) / "
            "추천 메뉴가 뭐예요? (What do you recommend?) / "
            "맵지 않게 해 주세요 (Please make it not spicy)"
        ),
        metadata={"topic": "음식", "subtopic": "식당"},
    ),
    Document(
        page_content=(
            "맛 표현: 정말 맛있어요! (It's really delicious!) / "
            "좀 짜요 (It's a bit salty) / "
            "달콤하고 부드러워요 (It's sweet and smooth)"
        ),
        metadata={"topic": "음식", "subtopic": "맛표현"},
    ),
    Document(
        page_content=(
            "요리 표현: 요리하는 것을 좋아해요 (I like cooking) / "
            "어떻게 만들어요? (How do you make it?) / "
            "재료가 뭐예요? (What are the ingredients?)"
        ),
        metadata={"topic": "음식", "subtopic": "요리"},
    ),
    # ── 날씨 (Weather) ─────────────────────────────────────────────
    Document(
        page_content=(
            "날씨 표현: 오늘 날씨가 어때요? (How is the weather today?) / "
            "비가 올 것 같아요 (It looks like rain) / "
            "많이 춥네요 (It's really cold)"
        ),
        metadata={"topic": "날씨", "subtopic": "표현"},
    ),
    Document(
        page_content=(
            "계절 표현: 봄에는 꽃이 많이 펴요 (Flowers bloom a lot in spring) / "
            "여름은 너무 더워요 (Summer is too hot) / "
            "가을에는 단풍이 예뻐요 (Autumn leaves are beautiful) / "
            "겨울에는 눈이 와요 (It snows in winter)"
        ),
        metadata={"topic": "날씨", "subtopic": "계절"},
    ),
    # ── 직업 (Work) ────────────────────────────────────────────────
    Document(
        page_content=(
            "직업 소개: 무슨 일을 해요? (What do you do?) / "
            "저는 회사원이에요 (I'm an office worker) / "
            "어디서 일해요? (Where do you work?)"
        ),
        metadata={"topic": "직업", "subtopic": "소개"},
    ),
    Document(
        page_content=(
            "회사 생활: 출퇴근 시간이 어떻게 돼요? (What are your commuting hours?) / "
            "야근이 많아요? (Do you often work overtime?) / "
            "회의가 있어요 (I have a meeting)"
        ),
        metadata={"topic": "직업", "subtopic": "회사생활"},
    ),
    # ── 가족 (Family) ──────────────────────────────────────────────
    Document(
        page_content=(
            "가족 소개: 가족이 몇 명이에요? (How many people are in your family?) / "
            "형제자매가 있어요? (Do you have siblings?) / "
            "부모님이 어디에 사세요? (Where do your parents live?)"
        ),
        metadata={"topic": "가족", "subtopic": "소개"},
    ),
    Document(
        page_content=(
            "가족 호칭: 아버지/아빠 (father), 어머니/엄마 (mother), "
            "형/오빠 (older brother), 누나/언니 (older sister), "
            "남동생/여동생 (younger sibling), 할아버지/할머니 (grandparents)"
        ),
        metadata={"topic": "가족", "subtopic": "호칭"},
    ),
]


def get_vector_store() -> Chroma:
    """ChromaDB 인스턴스를 반환. 데이터가 없으면 샘플 문서를 초기 로드."""
    embeddings = OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    vector_store = Chroma(
        collection_name="korean_phrases",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # 컬렉션이 비어 있을 때만 샘플 데이터 삽입
    if len(vector_store.get(limit=1)["ids"]) == 0:
        print(f"[ChromaDB] 샘플 한국어 표현 {len(KOREAN_PHRASES)}개를 초기 로드합니다...")
        vector_store.add_documents(KOREAN_PHRASES)

    return vector_store


def get_retriever(k: int = 3):
    """주어진 쿼리에 가장 유사한 문서 k개를 반환하는 retriever."""
    return get_vector_store().as_retriever(search_kwargs={"k": k})
