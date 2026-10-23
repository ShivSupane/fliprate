"""Shared language rules: which romanized spellings are safe to swap, and which are NOT.

Design principle
----------------
Only transformations that provably keep the meaning are called "safe variants".
Anything ambiguous is left alone and reported as a warning instead of being silently rewritten.

Language coverage (romanized / Latin-script social-media typing)
----------------------------------------------------------------
  Hindi/Hinglish, Urdu (Roman Urdu), Marathi, Bengali (Banglish),
  Tamil (Tanglish), Telugu (Tenglish), Kannada (Kanglish),
  Malayalam (Manglish), Gujarati, Punjabi, Odia, Assamese.

Confidence markers
------------------
  # HIGH   — well-documented in multiple published linguistic sources
  # MEDIUM — consistent with phonology and common usage; speaker review recommended
  # LOW    — phonology-based draft only; MUST be reviewed by a fluent speaker before production

How to use
----------
  from fliprate.rules import DEFAULT_RULES               # core HIGH groups, all languages

  # Add one language on top of defaults
  from fliprate.rules import LANGUAGE_GROUPS, DEFAULT_RULES
  rules = DEFAULT_RULES.extended(groups=LANGUAGE_GROUPS["tamil"])

  # Build a custom set
  from fliprate.rules import CORE_GROUPS, LANGUAGE_GROUPS, RuleSet
  rules = RuleSet(groups=CORE_GROUPS + LANGUAGE_GROUPS["telugu"] + LANGUAGE_GROUPS["kannada"])

  # All languages (includes MEDIUM/LOW — use with caution)
  rules = RuleSet(groups=sum(LANGUAGE_GROUPS.values(), []))

Sources (content paraphrased for compliance)
--------------------------------------------
  - dhvani library (github.com/Kkoundinyaa/dhvani): 1M+ Hindi spelling variant lexicon,
    documents bahut/bohot/boht/bhot/bahot, accha/achha/acha/achaa, kaise/kese/kayse
  - ACL 2025 IndoNLP (aclanthology.org): namaste/namastey/namste/nmst Hindi variants
  - SikhiWiki Roman Punjabi (sikhiwiki.org): tussi/tusi, kive/kiwe, sat sri akal
  - Wikibooks Gujarati common phrases: tame kem chho / tu kem chhe / tu kya che
  - Wikivoyage Malayalam phrasebook: full Manglish spelling reference (illa/alla, venda,
    njaan/njan, cheyyunnu, undo, entha/enthu)
  - FSI Bengali course: ami bhalo achhi, apni kemon achhen
  - Quora: bhalo/valo Bengali (pronunciation-based), ami tomake/tomay (same pronoun)
  - Quora: chala bagundi / chala manchidi Telugu variants
  - Washington University sociolinguistics paper on Roman Hindi-Urdu orthographic conventions
  - italki.com: nahi/nahe/nahin are the same Urdu word
  - University of Chicago Berntsen Marathi dictionary (dsal.uchicago.edu)
  - MTSU Tamil vocabulary resource (cala.mtsu.edu)
  - mykannadateacher.wordpress.com: naanu/nanu, neevu/nivu, illa, howdu basic Kannada
  - Scribd Telugu-English-Hindi guide: nuvvu/meeru/manamu/eppudu/undi
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple


# =============================================================================
# 1. HINDI / HINGLISH
#    Primary source: dhvani library (1M+ lexicon), ACL 2025 IndoNLP paper,
#    Washington University Roman Hindi-Urdu orthographic conventions paper.
#    Key finding: "bahut/bohot/boht/bhot" and "accha/achha/acha/achaa" are the
#    canonical examples in every published Hinglish normalization study.
# =============================================================================
HINDI_GROUPS: List[Tuple[str, ...]] = [
    # ── Negation ──────────────────────────────────────────────────────────────
    ("nahi", "nhi", "nahin", "nahee", "nhe"),        # HIGH — most-studied Hinglish variant
    ("mat", "maat"),                                  # HIGH — imperative negation "don't"

    # ── Degree / quantity ─────────────────────────────────────────────────────
    ("bahut", "bohot", "bahot", "bhut", "boht", "bhot"),  # HIGH — dhvani canonical example
    ("thoda", "thora", "thoda"),                      # HIGH — "a little"
    ("zyada", "jyada", "zyaada", "jyaada"),           # HIGH — "more/too much"
    ("bilkul", "bilkul", "blkul"),                    # HIGH — "absolutely/completely"
    ("sirf", "srf"),                                  # HIGH — "only"
    ("kabhi", "kbhi", "kbi"),                         # MEDIUM — "ever/sometimes"
    ("jaldi", "jldi"),                                # HIGH — "quickly"
    ("zaroor", "zarur", "zarurr"),                    # HIGH — "definitely/certainly"
    ("shayad", "shayd"),                              # MEDIUM — "maybe"
    ("phir", "fir"),                                  # HIGH — "then/again"

    # ── Common words ──────────────────────────────────────────────────────────
    ("accha", "acha", "achha", "achchha", "achaa"),   # HIGH — dhvani canonical example
    ("theek", "thik", "thek", "teek"),                # HIGH — "okay/correct"
    ("kyun", "kyu", "kyon"),                          # HIGH — "why"
    ("kya", "kyaa"),                                  # HIGH — "what"
    ("kaisa", "kaysa", "kesa", "kaissa"),             # HIGH — "how/what kind" (dhvani: kaise/kese/kayse)
    ("kuch", "kutch", "kucch"),                       # HIGH — "something/some"
    ("baat", "bat"),                                  # HIGH — "talk/matter"
    ("woh", "wo", "voh", "vo"),                       # HIGH — "that/he/she"
    ("yeh", "ye"),                                    # HIGH — "this/these"
    ("abhi", "abi"),                                  # HIGH — "right now"
    ("bhi", "bhee"),                                  # HIGH — "also/too"
    ("samajh", "smjh"),                               # MEDIUM — "understand"
    ("lekin", "lakin", "lkn"),                        # HIGH — "but"
    ("aur", "or"),                                    # MEDIUM — "and" (keep MEDIUM: or=English)

    # ── Pronouns ──────────────────────────────────────────────────────────────
    ("main", "mai", "mein"),                          # HIGH — "I"
    ("mujhe", "mujhay", "mujhy"),                     # HIGH — "to me"
    ("aap", "ap"),                                    # HIGH — "you (formal)"
    ("koi", "koyi"),                                  # MEDIUM — "someone/any"

    # ── Verbs ─────────────────────────────────────────────────────────────────
    ("karna", "krna"),                                # HIGH — "to do"
    ("karo", "kro"),                                  # HIGH — "do (informal)"
    ("karein", "kare", "karen"),                      # HIGH — "please do"
    ("gaya", "gya", "gayaa"),                         # HIGH — "went/gone"
    ("hua", "hoa"),                                   # HIGH — "happened/became"
    ("raha", "rha", "rhaa"),                          # HIGH — progressive aux (masc)
    ("rahi", "rhi", "rhii"),                          # HIGH — progressive aux (fem)
    ("batao", "btao"),                                # MEDIUM — "tell me"
    ("suno", "sno"),                                  # MEDIUM — "listen"
    ("bolna", "blna"),                                # MEDIUM — "to speak"
    ("milna", "mlna"),                                # MEDIUM — "to meet/receive"
    ("aana", "ana"),                                  # MEDIUM — "to come"
    ("jana", "jnaa"),                                 # MEDIUM — "to go"

    # ── Greetings / courtesy ──────────────────────────────────────────────────
    # ACL 2025 paper documents: "Namaste appearing as Namste, Nmst, or Namastey"
    ("namaste", "namaskar", "namastey", "namste", "nmst"),  # HIGH — ACL 2025 documented
    ("shukriya", "shukria", "sukria", "shukriyah"),   # HIGH — "thank you"
    ("please", "plz", "pls"),                         # HIGH — loanword

    # ── Customer support ──────────────────────────────────────────────────────
    ("wrong", "rong", "wrng"),                        # MEDIUM
    ("return", "retrun", "rtrn"),                     # MEDIUM
    ("complaint", "complnt", "cmplnt"),               # MEDIUM
    ("support", "supprt", "sprt"),                    # MEDIUM
    ("tracking", "trcking"),                          # MEDIUM
    ("damaged", "damgd"),                             # MEDIUM
    ("courier", "courir", "curior"),                  # MEDIUM
    ("parcel", "parcl"),                              # MEDIUM
    ("agent", "agnt"),                                # MEDIUM
    ("manager", "managr", "mngr"),                    # MEDIUM
]

# =============================================================================
# 2. URDU / ROMAN URDU
#    Primary source: italki.com "nahi/nahe/nahin are the same word",
#    ACL EMNLP 2015 paper on lexical variations ("zindagee, zindagy, zaindagee, zndagi"),
#    Washington University Roman Hindi-Urdu paper (shared Hindustani base).
#    Note: Hindi and Urdu share most of their spoken vocabulary (Hindustani base).
#    These groups cover Urdu-specific spellings and register differences.
# =============================================================================
URDU_GROUPS: List[Tuple[str, ...]] = [
    # ── Negation ──────────────────────────────────────────────────────────────
    ("nahi", "nahin", "naheen", "nahe", "nhi"),       # HIGH — italki documented
    ("mat", "maat"),                                  # HIGH

    # ── Copula / auxiliary ────────────────────────────────────────────────────
    ("hai", "hei", "hy"),                             # HIGH — "is"
    ("hain", "hein", "hen"),                          # HIGH — "are (formal/plural)"
    ("tha", "thaa"),                                  # HIGH — "was" (masc)
    ("thi", "thee"),                                  # HIGH — "was" (fem)
    ("hoga", "hogaa"),                                # MEDIUM — "will be" (masc)
    ("hogi", "hogee"),                                # MEDIUM — "will be" (fem)

    # ── Function words ────────────────────────────────────────────────────────
    ("kya", "kia", "kyaa"),                           # HIGH — "what"
    ("kyun", "kyo", "kyon"),                          # HIGH — "why"
    ("kaise", "kese", "kayse"),                       # HIGH — "how" (dhvani documented)
    ("kaun", "kon"),                                  # HIGH — "who"
    ("lekin", "lakin", "magar", "lkn"),               # HIGH — "but/however"
    ("bhi", "bhee", "bi"),                            # HIGH — "also/too"
    ("sirf", "srf"),                                  # HIGH — "only"
    ("bilkul", "blkul"),                              # HIGH — "absolutely"
    ("zaroor", "zarur"),                              # HIGH — "definitely"
    ("abhi", "abi"),                                  # HIGH — "right now"
    ("pehle", "pehly", "pehley"),                     # HIGH — "before/first"
    ("kuch", "kch"),                                  # HIGH — "something"
    ("sab", "sb"),                                    # HIGH — "all/everyone"
    ("baad", "bad"),                                  # MEDIUM — "after/later"
    ("waqt", "waqat", "wakt"),                        # MEDIUM — "time"
    ("log", "loog"),                                  # MEDIUM — "people"
    ("ghar", "ghr"),                                  # MEDIUM — "home"

    # ── Courtesy ──────────────────────────────────────────────────────────────
    ("shukriya", "shukria", "sukria"),                # HIGH
    ("meherbani", "mehrbani", "meharbani"),            # HIGH — "thanks/kindness"
    ("inshallah", "inshaallah", "insha allah"),        # MEDIUM — "God willing"

    # ── Verbs ─────────────────────────────────────────────────────────────────
    ("karna", "krna"),                                # HIGH
    ("karo", "kro"),                                  # HIGH
    ("baat", "bat"),                                  # HIGH — "talk/matter"
    ("dekhna", "dkhna"),                              # MEDIUM — "to see"
    ("sunna", "snna"),                                # MEDIUM — "to listen"
    ("chahta", "chahtta"),                            # MEDIUM — "wants" (masc)
    ("chahti", "chahtee"),                            # MEDIUM — "wants" (fem)
    # ACL paper example: zindagee/zindagy/zaindagee/zndagi — shows vowel/consonant dropping
    ("zindagi", "zindagee", "zindagy", "zndagi"),     # HIGH — ACL 2015 canonical example
]

# =============================================================================
# 3. MARATHI
#    Primary source: University of Chicago Berntsen Marathi dictionary,
#    Springer paper on Roman Marathi transliteration (2023),
#    SentiMix Marathi dataset spellings.
#    Key: Marathi uses verb agreement for gender (masc/fem suffix pairs).
# =============================================================================
MARATHI_GROUPS: List[Tuple[str, ...]] = [
    # ── Negation ──────────────────────────────────────────────────────────────
    ("nako", "nko", "nakoo"),                         # HIGH — "don't want / no"
    ("nahi", "nhi", "nahin"),                         # HIGH — shared with Hindi

    # ── Copula / auxiliary ────────────────────────────────────────────────────
    ("aahe", "ahe"),                                  # HIGH — "is" (present)
    ("aahet", "ahet"),                                # HIGH — "are" (plural)
    ("hota", "hotta"),                                # HIGH — "was" (masc)
    ("hoti", "hotti"),                                # HIGH — "was" (fem)
    ("jhala", "zala"),                                # HIGH — "happened/became" (masc; zh/z both used)
    ("jhali", "zali"),                                # HIGH — "happened/became" (fem)
    ("gela", "gla"),                                  # HIGH — "went" (masc)
    ("geli", "gli"),                                  # HIGH — "went" (fem)
    ("aala", "ala"),                                  # HIGH — "came" (masc)
    ("aali", "ali"),                                  # HIGH — "came" (fem)
    ("hoil", "hoeel"),                                # MEDIUM — "will be"
    ("sangto", "sangto"),                             # MEDIUM — "I tell" (masc)
    ("sangti", "sangti"),                             # MEDIUM — "I tell" (fem)

    # ── Pronouns ──────────────────────────────────────────────────────────────
    ("mala", "malaa"),                                # HIGH — "to me"
    ("tula", "tulaa"),                                # HIGH — "to you (informal)"
    ("aplya", "aaplya"),                              # MEDIUM — "our"
    ("tumhi", "tumhee"),                              # HIGH — "you (plural)"
    ("tumhala", "tumhla"),                            # MEDIUM — "to you (formal)"

    # ── Adverbs / connectors ──────────────────────────────────────────────────
    ("khup", "khoop"),                                # HIGH — "very"
    ("ajun", "ajoon", "azun"),                        # HIGH — "still/yet"
    ("ata", "aata"),                                  # HIGH — "now"
    ("jar", "jara"),                                  # MEDIUM — "if"
    ("tar", "tara"),                                  # MEDIUM — "then/so"
    ("mhanaje", "mhanje"),                            # MEDIUM — "meaning/that is"
    ("sagla", "sagale", "sagle"),                     # MEDIUM — "all"
    ("bara", "barra"),                                # MEDIUM — "okay/right" (discourse marker)

    # ── Customer-support Marathi ──────────────────────────────────────────────
    ("avadla", "avdla"),                              # HIGH — "liked" (masc)
    ("avadli", "avdli"),                              # HIGH — "liked" (fem)
    ("milala", "milela"),                             # HIGH — "received/got" (masc)
    ("milali", "mileli"),                             # HIGH — "received/got" (fem)
    ("pohchala", "pohochala", "pohcla"),              # HIGH — "arrived/reached"
    ("pathavla", "pathavlaa"),                        # MEDIUM — "sent" (masc)
    ("dhanyawad", "dhanyavaad", "dhnyvad"),           # HIGH — "thank you"
    ("kara", "kra"),                                  # MEDIUM — "do" (imperative)
    ("dya", "daa"),                                   # MEDIUM — "give" (imperative)
    ("ghya", "ghyaa"),                                # MEDIUM — "take" (imperative)
    ("thamba", "thambaa", "thambava"),                # MEDIUM — "stop/wait"
    ("pahije", "pahijey"),                            # HIGH — "need/want"
    ("bhari", "bharri"),                              # MEDIUM — "awesome/great" (Marathi slang, Quora)
]

# =============================================================================
# 4. BENGALI / BANGLISH
#    Primary source: FSI Bengali course ("ami bhalo achhi", "apni kemon achhen?"),
#    gamitisa.com Bangla-to-Banglish converter,
#    Quora: bhalo vs valo ("matching the pronunciation rather than the script"),
#    Quora: ami tomake/tomay bhalobashi = same meaning (same objective pronoun),
#    HuggingFace bangla-transliteration model: "both bhalo and valo for ভালো,
#    reflecting how different people actually type the same word".
#    Bengali has 3 levels of "you": apni (formal), tumi (familiar), tui (intimate).
# =============================================================================
BENGALI_GROUPS: List[Tuple[str, ...]] = [
    # ── Negation ──────────────────────────────────────────────────────────────
    ("naa", "na"),                                    # HIGH — "no/not" (naa=canonical to avoid AMBIGUOUS "na")
    ("noy", "noi"),                                   # MEDIUM — "is not" (verb negation)
    ("nei", "nei"),                                   # MEDIUM — "not there" (existential negation)

    # ── Pronouns ──────────────────────────────────────────────────────────────
    ("ami", "aami"),                                  # HIGH — "I" (FSI documented)
    ("apni", "aapni"),                                # HIGH — "you (formal)" (FSI documented)
    ("tumi", "toomi"),                                # HIGH — "you (familiar)" (FSI documented)
    ("tui", "tooi"),                                  # MEDIUM — "you (intimate)" (Quora: social media uses tui)
    # Quora: "ami tomake bhalobashi / ami tomay bhalobashi = same meaning"
    ("tomake", "tomay"),                              # HIGH — objective "you (familiar)" — documented Quora
    ("amader", "aamader"),                            # MEDIUM — "our"

    # ── Copula / auxiliary ────────────────────────────────────────────────────
    # FSI: "ami bhalo achhi" — achi/achhi standard Banglish spellings
    ("achi", "achhi"),                                # HIGH — "am/is/are" (1st/2nd) — FSI canonical
    # FSI: "apni kemon achhen?" — ache/achhe standard
    ("ache", "achhe"),                                # HIGH — "is/are" (3rd person) — FSI canonical
    ("chhilo", "chilo"),                              # MEDIUM — "was/were"
    ("hobe", "hobbe"),                                # MEDIUM — "will be"
    ("hoyeche", "hoeche"),                            # MEDIUM — "has happened/become"
    ("giyeche", "geche"),                             # MEDIUM — "has gone"

    # ── Common adjectives / adverbs ───────────────────────────────────────────
    # Quora + HuggingFace model: "bhalo or valo, matching pronunciation rather than script"
    ("bhalo", "valo"),                                # HIGH — "good" — documented from two sources
    ("kharap", "khrap"),                              # MEDIUM — "bad"
    ("beshi", "besi"),                                # MEDIUM — "more/a lot"
    ("onek", "anek"),                                 # MEDIUM — "many/a lot"
    ("shundor", "sundor", "sunder"),                  # MEDIUM — "beautiful"
    ("thik", "theek"),                                # MEDIUM — "correct/fine" (same word as Hindi theek)

    # ── Greetings / courtesy ──────────────────────────────────────────────────
    # FSI: "apni kemon achhen?" — kemon = standard
    ("kemon", "kmon"),                                # HIGH — "how" — FSI documented
    ("dhonnobad", "dhanyabad", "dhonnybad"),          # HIGH — "thank you"

    # ── Verbs ─────────────────────────────────────────────────────────────────
    ("koro", "kro"),                                  # MEDIUM — "do (imperative)"
    ("dao", "daw"),                                   # MEDIUM — "give (imperative)"
    ("jao", "jaw"),                                   # MEDIUM — "go (imperative)"
    ("bolo", "blo"),                                  # MEDIUM — "say (imperative)"
    ("dekho", "dkho"),                                # MEDIUM — "look (imperative)"
    ("shono", "sono"),                                # MEDIUM — "listen (imperative)"
    ("chai", "chhai"),                                # MEDIUM — "want"
    ("peyechi", "peychi"),                            # MEDIUM — "I have received"
    ("pathao", "pathav"),                             # MEDIUM — "send (imperative)"
    ("asbe", "ashbe"),                                # MEDIUM — "will come"
    ("jabe", "jabe"),                                 # MEDIUM — "will go"
]

# =============================================================================
# 5. TAMIL / TANGLISH
#    Primary source: Wikipedia:Tanglish, MTSU Tamil vocabulary (cala.mtsu.edu),
#    preply.com Tamil greetings ("vanakkam" = most common greeting documented),
#    loremipsumtext.com Tamil typing ("vanakkam, nandri, naan, enna are built in"),
#    tamilarticless.wordpress.com: I=Naan, Me=Ennai, We=Naangal, My=En/Enn.
#    Key: Tamil has three levels of formality for "you": nee (intimate),
#    neenga (polite/plural), neengal (formal).
# =============================================================================
TAMIL_GROUPS: List[Tuple[str, ...]] = [
    # ── Negation ──────────────────────────────────────────────────────────────
    ("illa", "illai"),                                # HIGH — "no/not there" (main negation)
    ("vendam", "venda", "vendaam"),                   # HIGH — "don't want/no need"
    ("mudiyadu", "mudiyathu"),                        # MEDIUM — "cannot/not possible"

    # ── Greetings ─────────────────────────────────────────────────────────────
    # preply + loremipsumtext: "vanakkam" is the standard greeting, typed many ways
    ("vanakkam", "vanakam"),                          # HIGH — "hello/greeting"
    ("nandri", "nandree"),                            # HIGH — "thank you"
    ("romba", "rombaa"),                              # HIGH — "very much" (emphatic degree)
    ("nalla", "nala"),                                # HIGH — "good"
    ("seri", "sery"),                                 # HIGH — "okay/alright"

    # ── Pronouns ──────────────────────────────────────────────────────────────
    # MTSU / tamilarticless: Naan=I, Nee=you, Neenga=you(polite)
    ("naan", "nan"),                                  # HIGH — "I" (tamilarticless documented)
    ("nee", "ni"),                                    # HIGH — "you (intimate)"
    ("neenga", "ninga"),                              # HIGH — "you (formal/plural)"
    ("naangal", "nangal"),                            # MEDIUM — "we (exclusive)"
    ("namma", "nama"),                                # MEDIUM — "our/we (inclusive)"
    ("avan", "aven"),                                 # MEDIUM — "he"
    ("aval", "avel"),                                 # MEDIUM — "she"
    ("avanga", "avarga"),                             # MEDIUM — "they"
    ("ennai", "enai"),                                # MEDIUM — "me" (tamilarticless: ennai)

    # ── Common question words ─────────────────────────────────────────────────
    ("enna", "yenna"),                                # HIGH — "what" (loremipsumtext documented)
    ("eppo", "yeppo"),                                # MEDIUM — "when"
    ("enga", "yenga"),                                # MEDIUM — "where"
    ("eppadi", "yeppadi"),                            # MEDIUM — "how"
    ("yean", "yen"),                                  # MEDIUM — "why" (yen=ambiguous, yean=clearer)

    # ── Degree words ──────────────────────────────────────────────────────────
    ("konjam", "konjum"),                             # HIGH — "a little"
    ("innum", "innam"),                               # MEDIUM — "still/more"

    # ── Copula / auxiliary ────────────────────────────────────────────────────
    ("irukku", "iruku"),                              # HIGH — "is/there is" (different keys — valid group)
    ("aachu", "achu"),                                # MEDIUM — "it happened/done"

    # ── Verbs ─────────────────────────────────────────────────────────────────
    ("sollu", "solu"),                                # MEDIUM — "say/tell"
    ("paaru", "paru"),                                # MEDIUM — "look/see"
    ("theriyum", "theriam"),                          # MEDIUM — "will know"
    ("puriyala", "purila"),                           # MEDIUM — "didn't understand"
    ("pannunga", "pannu"),                            # MEDIUM — "do (polite/imper.)"
    ("vaa", "va"),                                    # MEDIUM — "come (imperative)"
    ("poda", "pota"),                                 # MEDIUM — "go (away)" — informal
]

# =============================================================================
# 6. TELUGU / TENGLISH
#    Primary source: Wikipedia:Tenglish, Scribd Telugu-English-Hindi guide
#    (nuvvu/meeru/manamu/eppudu/undi documented with Hindi equivalents),
#    Quora: "chala bagundi" and "chala manchidi" (both = "very nice"),
#    Quora: "nenu bagunnanu = I am fine".
# =============================================================================
TELUGU_GROUPS: List[Tuple[str, ...]] = [
    # ── Negation ──────────────────────────────────────────────────────────────
    ("ledu", "ledhu"),                                # HIGH — "is not/there is not"
    ("vaddu", "vaddhu"),                              # HIGH — "don't/no"
    ("kaadu", "kaadhu"),                              # HIGH — "is not (copula negation)"

    # ── Greetings ─────────────────────────────────────────────────────────────
    ("namaskaram", "namskaram"),                      # HIGH — "hello/respect"
    ("dhanyavaadalu", "dhanyavadalu"),                # HIGH — "thank you (Telugu)"
    # Quora: "chala bagundi" and "chala manchidi" both well-documented
    ("chala", "chaala"),                              # HIGH — "very/a lot"
    ("bagundi", "baagundi"),                          # HIGH — "is good"
    ("manchidi", "manchdi"),                          # HIGH — "is good/nice"
    ("bagunnanu", "bagunanu"),                        # MEDIUM — "I am fine" (Quora: nenu bagunnanu)

    # ── Pronouns ──────────────────────────────────────────────────────────────
    # Scribd: nuvvu=you(informal), meeru=you(formal/plural), manamu=we
    ("nenu", "nennu"),                                # HIGH — "I"
    ("nuvvu", "nuvu"),                                # HIGH — Scribd documented
    ("meeru", "meru"),                                # HIGH — Scribd documented
    ("manamu", "manamu"),                             # MEDIUM — Scribd: "we" — same key, keep as-is
    ("mana", "maana"),                                # MEDIUM — "our/we (inclusive)"

    # ── Copula / auxiliary ────────────────────────────────────────────────────
    ("undi", "undy"),                                 # HIGH — "is/there is"
    ("ayyindi", "ayindi"),                            # HIGH — "happened/became"
    ("kavali", "kaavali"),                            # HIGH — "need/want"
    ("chesanu", "chesaanu"),                          # MEDIUM — "I did"
    ("vastundi", "vasthundi"),                        # MEDIUM — "will come"

    # ── Common words ──────────────────────────────────────────────────────────
    # Scribd: eppudu=when, ippudu=now documented
    ("ippudu", "ippude"),                             # MEDIUM — "now"
    ("eppudu", "epudu"),                              # MEDIUM — "when/always" (Scribd)
    ("kuda", "kooda"),                                # HIGH — "also/too"
    ("ikkade", "ikade"),                              # MEDIUM — "here"
    ("akkade", "akade"),                              # MEDIUM — "there"
    ("anta", "antaa"),                                # MEDIUM — "like that/quotative"

    # ── Verbs ─────────────────────────────────────────────────────────────────
    ("cheyyi", "cheyi"),                              # HIGH — "do (imperative)"
    ("chuddu", "chudu"),                              # MEDIUM — "see/look"
    ("vinnu", "vinu"),                                # MEDIUM — "listen"
    ("cheppu", "chepu"),                              # MEDIUM — "tell/say"
]

# =============================================================================
# 7. KANNADA / KANGLISH
#    Primary source: mykannadateacher.wordpress.com (naanu/nanu, neevu/nivu,
#    avanu/avnu, avalu/avlu, avaru/avru, naavu/navu — all documented with examples),
#    Wikipedia:Kanglish, Scribd Kannada pronouns guide.
#    Key: Kannada has a well-documented three-way pronoun system.
# =============================================================================
KANNADA_GROUPS: List[Tuple[str, ...]] = [
    # ── Negation ──────────────────────────────────────────────────────────────
    ("illa", "illaa"),                                # HIGH — "no/not there" (mykannadateacher)
    ("beda", "beeda"),                                # HIGH — "don't want/no"
    ("agalla", "aagalla"),                            # MEDIUM — "won't happen"
    ("maadalla", "madalla"),                          # MEDIUM — "won't do"

    # ── Greetings ─────────────────────────────────────────────────────────────
    ("namaskara", "namskara"),                        # HIGH — "hello/greetings"
    ("dhanyavada", "dhanyavaada"),                    # HIGH — "thank you"
    ("chennagide", "chenagide"),                      # HIGH — "is good/fine"
    ("summane", "sumne"),                             # MEDIUM — "just like that"

    # ── Pronouns ──────────────────────────────────────────────────────────────
    # mykannadateacher: naanu, neevu, avanu, avalu, avaru, naavu documented with sentences
    ("naanu", "nanu"),                                # HIGH — "I" (mykannadateacher)
    ("neevu", "nivu"),                                # HIGH — "you (formal)" (mykannadateacher)
    ("neenu", "ninu"),                                # HIGH — "you (informal)"
    ("avanu", "avnu"),                                # HIGH — "he" (mykannadateacher)
    ("avalu", "avlu"),                                # HIGH — "she" (mykannadateacher)
    ("avaru", "avru"),                                # HIGH — "they/he(respect)" (mykannadateacher)
    ("naavu", "navu"),                                # HIGH — "we" (mykannadateacher)
    ("nimma", "nima"),                                # MEDIUM — "your (formal)"
    ("nanna", "nna"),                                 # MEDIUM — "my"

    # ── Copula / auxiliary ────────────────────────────────────────────────────
    ("aayitu", "ayitu"),                              # HIGH — "it happened/done"
    ("bekku", "beku"),                                # HIGH — "need/want"
    ("ede", "ide"),                                   # MEDIUM — "is/there is (proximal)"
    ("irutte", "irute"),                              # MEDIUM — "will be/used to be"

    # ── Common words / verbs ──────────────────────────────────────────────────
    ("maadu", "madu"),                                # HIGH — "do/make"
    ("helu", "hellu"),                                # HIGH — "say/tell"
    ("nodu", "noodu"),                                # HIGH — "see/look"
    ("kodu", "koodu"),                                # MEDIUM — "give"
    ("hege", "heg"),                                  # HIGH — "how"
    ("yake", "yaake"),                                # MEDIUM — "why"
    ("yelli", "elli"),                                # MEDIUM — "where"
    ("kooda", "kuda"),                                # HIGH — "also/too"
    ("anta", "antha"),                                # MEDIUM — "like that"
    ("hogbeku", "hogbku"),                            # MEDIUM — "should go" (talkpal.ai)
    ("howdu", "houdu"),                               # HIGH — "yes/correct" (mykannadateacher)
]

# =============================================================================
# 8. MALAYALAM / MANGLISH
#    Primary source: Wikivoyage Malayalam phrasebook (full Manglish reference),
#    which explicitly documents: njaan/njan=I, nee/ni=you(informal),
#    ningal=you(formal/plural), illa=no, alla=is not, venda=don't want,
#    undo=is there?, cheyyunnu=is doing, entha=what, eppo=when.
#    Also: "Yes=athey/sheri, No=illa/alla, Thank you=nanni".
# =============================================================================
MALAYALAM_GROUPS: List[Tuple[str, ...]] = [
    # ── Negation ──────────────────────────────────────────────────────────────
    ("illa", "ila"),                                  # HIGH — "no/not there" (Wikivoyage)
    ("alla", "ala"),                                  # HIGH — "is not" (Wikivoyage)
    ("venda", "vendaa"),                              # HIGH — "don't want/no" (Wikivoyage)
    ("ariyilla", "ariyila"),                          # MEDIUM — "don't know"

    # ── Affirmative ───────────────────────────────────────────────────────────
    ("athey", "athe"),                                # HIGH — "yes/that's it" (Wikivoyage: athey)
    ("sheri", "seri"),                                # HIGH — "okay/correct" (Wikivoyage: sheri)

    # ── Greetings / courtesy ──────────────────────────────────────────────────
    ("namaskaram", "namskaram"),                      # HIGH — "hello (formal)" (Wikivoyage)
    ("nanni", "nandri"),                              # HIGH — "thank you" (Wikivoyage: nanni)
    ("santhosham", "santosham"),                      # MEDIUM — "happy/happiness"

    # ── Question words ────────────────────────────────────────────────────────
    # Wikivoyage documents "entha karyam" and "entha preshnam" (what's the matter)
    ("entha", "enta"),                                # HIGH — "what" (Wikivoyage documented)
    ("eppo", "yepo"),                                 # MEDIUM — "when"
    ("evide", "evde"),                                # MEDIUM — "where" (Wikivoyage: evideyanu?)
    ("engane", "engne"),                              # MEDIUM — "how"

    # ── Pronouns ──────────────────────────────────────────────────────────────
    # Wikivoyage: njaan=I, nee=you(informal), ningal=you(formal/plural)
    ("njaan", "njan"),                                # HIGH — "I" (Wikivoyage)
    ("nee", "ni"),                                    # HIGH — "you (intimate)"
    ("ningal", "ningal"),                             # HIGH — "you (formal/plural)" (Wikivoyage)
    ("avan", "avn"),                                  # MEDIUM — "he"
    ("aval", "avl"),                                  # MEDIUM — "she"
    ("nammal", "namal"),                              # MEDIUM — "we (inclusive)"
    ("njangal", "nangal"),                            # MEDIUM — "we (exclusive)"

    # ── Copula / verbs ────────────────────────────────────────────────────────
    # Wikivoyage: "undo" = "is there?" documented explicitly
    ("undo", "und"),                                  # HIGH — "is there?" (Wikivoyage)
    ("aanu", "anu"),                                  # HIGH — "is/am" (Wikivoyage: sukhamANu)
    # Quora: "njaan joli cheyyunnu" = "I am working" — cheyyunnu documented
    ("cheyyunnu", "cheyyunu"),                        # HIGH — "is doing" (Quora)
    ("cheythu", "cheytu"),                            # HIGH — "did"
    ("cheyyum", "cheyum"),                            # MEDIUM — "will do"
    ("kitto", "kittu"),                               # MEDIUM — "will get/receive"
    ("aakum", "akum"),                                # MEDIUM — "will become"
    # Wikivoyage: "vellam evide kittum?" (where can I get water) — kittum documented
    ("kittum", "kitum"),                              # MEDIUM — "will get" (Wikivoyage)
    # Wikivoyage: "pinne kanam" (goodbye), "innu" (today), "innale" (yesterday)
    ("innu", "inu"),                                  # MEDIUM — "today"
    ("innale", "inale"),                              # MEDIUM — "yesterday"
    ("naale", "nale"),                                # MEDIUM — "tomorrow"
]

# =============================================================================
# 9. GUJARATI
#    Primary source: Wikibooks Gujarati common phrases (fully fetched):
#    "Tame kem chho?" (formal) vs "Tu kem chhe?" (informal/same age) — documented
#    "Maja ma" = "I'm fine", "aavjo" = goodbye, "ha" = yes, "na" = no.
#    Latinum substack: chhe/chhuṃ/chho/hato/hati/hashe documented conjugation.
#    Scribd "Learn Gujarati in 30 Days": hun, tame, tu documented.
# =============================================================================
GUJARATI_GROUPS: List[Tuple[str, ...]] = [
    # ── Negation ──────────────────────────────────────────────────────────────
    ("nathi", "naathi"),                              # HIGH — "is not/no"

    # ── Greetings ─────────────────────────────────────────────────────────────
    # Wikibooks: "Tame kem chho? (older) / Tu kem chhe? (same age)" — both documented
    ("kem", "khem"),                                  # HIGH — "how" (Wikibooks)
    ("chho", "cho", "chhe", "che"),                   # HIGH — copula variants (Wikibooks documented)
    # Wikibooks: "Maja ma" = I'm fine
    ("majama", "majma", "maja ma"),                   # HIGH — "fine/well" (Wikibooks)
    ("aavjo", "avjo"),                                # HIGH — "goodbye" (Wikibooks)

    # ── Pronouns ──────────────────────────────────────────────────────────────
    # Scribd: "I = Mein/hun" both documented
    ("hun", "hoon"),                                  # HIGH — "I" (Scribd + Wikibooks)
    ("tame", "tame"),                                 # HIGH — "you (formal)" (Wikibooks)
    ("tu", "too"),                                    # HIGH — "you (informal)"
    ("te", "teh"),                                    # HIGH — "he/she/it"

    # ── Copula ────────────────────────────────────────────────────────────────
    # Latinum: chhe=is, chhuṃ=am, chho=are, hato=was(m), hati=was(f), hashe=will be
    ("chhu", "chu"),                                  # HIGH — "am" (1st person) (Latinum)
    ("chhie", "chie"),                                # MEDIUM — "are" (1st plural)
    ("hatu", "hattu"),                                # HIGH — "was" (masc) (Latinum)
    ("hati", "hatti"),                                # HIGH — "was" (fem) (Latinum)
    ("hashe", "hase"),                                # HIGH — "will be" (Latinum: hashe)
    ("thase", "thasse"),                              # MEDIUM — "will happen"

    # ── Common words ──────────────────────────────────────────────────────────
    # Wikibooks: "Tamaru naam su chhe?" — shu/su documented as "what"
    ("shu", "su"),                                    # HIGH — "what" (Wikibooks)
    ("ane", "aane"),                                  # HIGH — "and"
    ("kyare", "kyaare"),                              # MEDIUM — "when"
    ("kyan", "kyaan"),                                # MEDIUM — "where"
    ("dhanyavaad", "dhanyavad"),                      # HIGH — "thank you"
    ("maafi", "mafi"),                                # MEDIUM — "sorry/excuse"

    # ── Verbs ─────────────────────────────────────────────────────────────────
    # Wikibooks: "Tame Mane maddad karso?" — karo documented
    ("karo", "kro"),                                  # MEDIUM — "do"
    ("aavo", "avo"),                                  # MEDIUM — "come"
    ("jao", "javo"),                                  # MEDIUM — "go"
    ("kaho", "kaho"),                                 # MEDIUM — "say/tell"
    # Wikibooks: "aapno aabhar" = thank you; "Mane madad ni jaroor chhe" = I need help
    ("jaroor", "jarur"),                              # MEDIUM — "need/necessary" (Wikibooks)
    # Wikibooks: "Ha" = yes, "Na" = no
    ("haa", "ha"),                                    # HIGH — "yes" (Wikibooks: Ha=yes)
]

# =============================================================================
# 10. PUNJABI / PUNGLISH
#     Primary source: SikhiWiki Roman Punjabi (fully fetched):
#     "tussi kiwen ho? or tussi keedah ho?" — kive/kiwe documented,
#     main/maa, oh, tu/tusi, pheer, janna, aana documented with examples.
#     Quora: "Tu Kiddan Ya or Tussi Kiddan Ya — Standard Punjabi = Doabi Punjabi"
# =============================================================================
PUNJABI_GROUPS: List[Tuple[str, ...]] = [
    # ── Negation ──────────────────────────────────────────────────────────────
    ("nahi", "nhi", "nahin"),                         # HIGH — shared with Hindi/Urdu

    # ── Greetings ─────────────────────────────────────────────────────────────
    ("sat sri akal", "satshri akal", "sat siri akal"),  # HIGH — SikhiWiki: "formal Sikh greeting"
    ("waheguru", "wahiguru"),                            # HIGH — SikhiWiki
    # SikhiWiki: "Tussi kiwen ho?" or "Tussi keedah ho?"
    ("kive", "kiwe", "kiwen"),                          # HIGH — "how" (SikhiWiki: kiwen)
    ("kiddan", "kidam", "keedah"),                      # HIGH — "hi/how are you" (SikhiWiki: keedah)

    # ── Pronouns ──────────────────────────────────────────────────────────────
    # SikhiWiki: "I, me = maa, manu", "You = tu, tusi", "He/She = oh"
    ("main", "mai", "maa"),                             # HIGH — "I" (SikhiWiki: maa=main)
    ("tussi", "tusi"),                                  # HIGH — "you (polite)" (SikhiWiki)
    ("tu", "too"),                                      # HIGH — "you (informal)"
    ("oh", "uh"),                                       # MEDIUM — "he/she/that" (SikhiWiki: oh)
    ("assi", "asi"),                                    # MEDIUM — "we"
    ("ohna", "ona"),                                    # MEDIUM — "they"

    # ── Copula / auxiliary ────────────────────────────────────────────────────
    # SikhiWiki: "Is = Haa, hai", "nahee hai = It's not"
    ("hai", "hain", "haa"),                             # HIGH — "is/are" (SikhiWiki: Haa/hai)
    ("tha", "ta"),                                      # MEDIUM — "was" (masc)
    ("thi", "ti"),                                      # MEDIUM — "was" (fem)
    ("hovega", "hoga"),                                 # MEDIUM — "will be"

    # ── Adverbs ───────────────────────────────────────────────────────────────
    # SikhiWiki: "then = pheer"
    ("pheer", "phir", "fer"),                           # HIGH — "then/again" (SikhiWiki: pheer)
    ("jaldi", "jldi"),                                  # HIGH — "quickly"
    ("pehlan", "pehle"),                                # MEDIUM — "before/first"

    # ── Verbs ─────────────────────────────────────────────────────────────────
    # SikhiWiki documents: janna (to go), aana (to come), khana (to eat), pinna (to drink)
    ("janna", "jana"),                                  # HIGH — "to go" (SikhiWiki: janna)
    ("aana", "ana"),                                    # HIGH — "to come" (SikhiWiki: aana)
    ("karo", "kro"),                                    # HIGH — "do"
    ("karna", "krna"),                                  # HIGH — "to do"
    ("aao", "ao"),                                      # HIGH — "come (imperative)"
    ("jao", "jao"),                                     # HIGH — "go (imperative)"
    ("dasso", "daso"),                                  # MEDIUM — "tell (imperative)"
    ("puchho", "pucho"),                                # MEDIUM — "ask"
    # SikhiWiki: "dubaara bolo ji = please repeat"
    ("dobara", "dubaara", "dubara"),                    # HIGH — "again/repeat" (SikhiWiki: dubaaraa)
    ("shukriya", "sukria"),                             # HIGH — "thank you"
    ("dhanyavaad", "dhanyavad"),                        # HIGH — "thank you"
]

# =============================================================================
# 11. ODIA
#     Primary source: Wikipedia:Standard_Odia (Standard vs Dialectal Roman column),
#     LoC Odia romanization table 2024, Quora: "Mote tume bhala laga / mu tumaku pasand kare".
#     NOTE: All LOW unless marked MEDIUM. Romanized Odia is rare on social media.
#     A fluent Odia speaker MUST review before production use.
# =============================================================================
ODIA_GROUPS: List[Tuple[str, ...]] = [
    # Negation
    ("nahi", "nahin"),                                # MEDIUM — shared negation
    ("nuhein", "nuhen"),                              # LOW — "is not"

    # Copula
    ("achi", "aachi"),                                # MEDIUM — "is/am/are"
    ("thila", "thila"),                               # LOW — "was"
    ("heila", "hela"),                                # LOW — "happened"
    ("jiba", "jibe"),                                 # LOW — "will go"
    ("aasiba", "asiba"),                              # LOW — "will come"

    # Pronouns
    # Quora: "Mote tume bhala laga" — mote=to me, tume=you documented
    ("mun", "mu"),                                    # MEDIUM — "I"
    ("tume", "tume"),                                 # MEDIUM — "you (informal)" (Quora)
    ("apana", "aapna"),                               # MEDIUM — "you (formal)"
    ("ame", "ame"),                                   # MEDIUM — "we"

    # Common
    # Quora: "bhala" = "good/like" documented
    ("bhala", "bhaala"),                              # MEDIUM — "good" (Quora)
    ("kana", "kana"),                                 # MEDIUM — "what"
    ("kemiti", "kemte"),                              # LOW — "how"
    ("dhanyabad", "dhanyavaad"),                      # MEDIUM — "thank you"
    ("namaskar", "namaskara"),                        # MEDIUM — "hello" (vocal.media)
]

# =============================================================================
# 12. ASSAMESE
#     Source: Wikipedia:Assamese_language, LoC Assamese romanization tables.
#     NOTE: All LOW unless marked MEDIUM. Romanized Assamese is rare.
#     A fluent Assamese speaker MUST review before production use.
# =============================================================================
ASSAMESE_GROUPS: List[Tuple[str, ...]] = [
    # Negation (nai excluded — conflicts with Hindi AMBIGUOUS "nai")
    ("nahay", "nahoi"),                               # LOW — "is not"
    ("nohoy", "nohou"),                               # LOW — "is not (alt)"

    # Copula
    ("ase", "aase"),                                  # LOW — "is/am/are"
    ("achil", "asil"),                                # LOW — "was"
    ("hol", "hola"),                                  # LOW — "happened"
    ("gol", "gola"),                                  # LOW — "went"

    # Pronouns
    ("moi", "moi"),                                   # MEDIUM — "I"
    ("tumi", "tomi"),                                 # MEDIUM — "you (informal)"
    ("apuni", "apuni"),                               # MEDIUM — "you (formal)"
    ("aami", "ami"),                                  # MEDIUM — "we"

    # Common
    ("bhaal", "bhal"),                                # MEDIUM — "good"
    ("xundor", "sundor"),                             # LOW — "beautiful" (x=sh in Assamese)
    ("dhanyabad", "dhonyobad"),                       # LOW — "thank you"
]

# =============================================================================
# SHARED / PAN-INDIC customer-support loanwords
# =============================================================================
SHARED_GROUPS: List[Tuple[str, ...]] = [
    ("payment", "paymnt", "pymnt"),
    ("refund", "rifund", "refnd"),
    ("delivery", "delivry", "delevery", "dlvry"),
    ("product", "prodct", "produkt"),
    ("quality", "qualty", "kwality"),
    ("order", "ordr"),
    ("cancel", "cancl", "cncl"),
    ("amount", "amnt"),
    ("contact", "contct", "cntct"),
    ("received", "recieved", "recvd"),
    ("problem", "problm", "prblm"),
    ("number", "nmbr", "numbr"),
    ("please", "plz", "pls"),
]

# =============================================================================
# LANGUAGE_GROUPS dict — for building custom RuleSets
# =============================================================================
LANGUAGE_GROUPS: Dict[str, List[Tuple[str, ...]]] = {
    "hindi":     HINDI_GROUPS,
    "urdu":      URDU_GROUPS,
    "marathi":   MARATHI_GROUPS,
    "bengali":   BENGALI_GROUPS,
    "tamil":     TAMIL_GROUPS,
    "telugu":    TELUGU_GROUPS,
    "kannada":   KANNADA_GROUPS,
    "malayalam": MALAYALAM_GROUPS,
    "gujarati":  GUJARATI_GROUPS,
    "punjabi":   PUNJABI_GROUPS,
    "odia":      ODIA_GROUPS,
    "assamese":  ASSAMESE_GROUPS,
    "shared":    SHARED_GROUPS,
}

# =============================================================================
# CORE_GROUPS — what DEFAULT_RULES uses.
# Only HIGH-confidence groups, carefully deduplicated.
# Word keys must be unique across all CORE groups.
# =============================================================================
_CORE: List[Tuple[str, ...]] = [
    # ── Hindi / Urdu HIGH ─────────────────────────────────────────────────────
    ("nahi", "nhi", "nahin", "nahee"),
    ("mat", "maat"),
    ("bahut", "bohot", "bahot", "bhut", "boht", "bhot"),
    ("accha", "acha", "achha"),
    ("kyun", "kyu", "kyon"),
    ("theek", "thik", "thek"),
    ("please", "plz", "pls"),
    ("abhi", "abi"),
    ("sirf", "srf"),
    ("jaldi", "jldi"),
    ("lekin", "lakin"),
    ("bhi", "bhee"),
    ("kuch", "kutch"),
    ("woh", "wo", "voh", "vo"),
    ("main", "mai", "mein"),
    ("gaya", "gya"),
    ("raha", "rha"),
    ("rahi", "rhi"),
    ("karna", "krna"),
    ("baat", "bat"),
    ("shukriya", "shukria"),
    ("namaste", "namaskar", "namastey", "namste", "nmst"),
    ("thoda", "thora"),
    ("zyada", "jyada"),
    ("phir", "fir"),
    ("zaroor", "zarur"),
    ("kaise", "kese", "kayse"),
    # ── Marathi HIGH ──────────────────────────────────────────────────────────
    ("nako", "nko"),
    ("aahe", "ahe"),
    ("aahet", "ahet"),
    ("hota", "hotta"),
    ("hoti", "hotti"),
    ("khup", "khoop"),
    ("ajun", "ajoon", "azun"),
    ("avadla", "avdla"),
    ("avadli", "avdli"),
    ("milala", "milela"),
    ("milali", "mileli"),
    ("pohchala", "pohochala", "pohcla"),
    ("tumhi", "tumhee"),
    ("dhanyawad", "dhanyavaad", "dhnyvad"),
    ("mala", "malaa"),
    ("sagla", "sagale", "sagle"),
    ("mhanaje", "mhanje"),
    ("jhala", "zala"),
    ("jhali", "zali"),
    ("aala", "ala"),
    ("aali", "ali"),
    # ── Bengali HIGH ──────────────────────────────────────────────────────────
    ("ami", "aami"),
    ("bhalo", "valo"),
    ("ache", "achhe"),
    ("kemon", "kmon"),
    ("dhonnobad", "dhanyabad"),
    ("tomake", "tomay"),
    # ── Tamil HIGH ────────────────────────────────────────────────────────────
    ("vanakkam", "vanakam"),
    ("nandri", "nandree"),
    ("nalla", "nala"),
    ("romba", "rombaa"),
    ("illa", "illai"),
    ("vendam", "venda"),
    ("seri", "sery"),
    ("naan", "nan"),
    ("konjam", "konjum"),
    ("enna", "yenna"),
    # ── Telugu HIGH ───────────────────────────────────────────────────────────
    ("namaskaram", "namskaram"),
    ("bagundi", "baagundi"),
    ("manchidi", "manchdi"),
    ("ledu", "ledhu"),
    ("kavali", "kaavali"),
    ("ayyindi", "ayindi"),
    ("cheyyi", "cheyi"),
    # ── Kannada HIGH ──────────────────────────────────────────────────────────
    ("namaskara", "namskara"),
    ("dhanyavada", "dhanyavaada"),
    ("chennagide", "chenagide"),
    ("naanu", "nanu"),
    ("bekku", "beku"),
    ("aayitu", "ayitu"),
    ("maadu", "madu"),
    ("hege", "heg"),
    ("howdu", "houdu"),
    # ── Malayalam HIGH ────────────────────────────────────────────────────────
    ("entha", "enta"),
    ("njaan", "njan"),
    ("undo", "und"),
    ("cheyyunnu", "cheyyunu"),
    ("aanu", "anu"),
    ("athey", "athe"),
    ("nanni", "nandri"),
    # ── Gujarati HIGH ─────────────────────────────────────────────────────────
    ("kem", "khem"),
    ("chho", "cho", "chhe", "che"),
    ("majama", "majma"),
    ("hun", "hoon"),
    ("shu", "su"),
    ("dhanyavaad", "dhanyavad"),
    ("ane", "aane"),
    ("chhu", "chu"),
    ("hatu", "hattu"),
    ("hati", "hatti"),
    ("aavjo", "avjo"),
    # ── Punjabi HIGH ──────────────────────────────────────────────────────────
    ("sat sri akal", "satshri akal"),
    ("tussi", "tusi"),
    ("kive", "kiwe", "kiwen"),
    ("kiddan", "kidam", "keedah"),
    # ── Shared customer support ───────────────────────────────────────────────
    ("payment", "paymnt", "pymnt"),
    ("refund", "rifund", "refnd"),
    ("delivery", "delivry", "delevery", "dlvry"),
    ("product", "prodct", "produkt"),
    ("quality", "qualty", "kwality"),
    ("order", "ordr"),
    ("cancel", "cancl", "cncl"),
    ("amount", "amnt"),
]

# =============================================================================
# DEVANAGARI SAFE GROUPS — curated meaning-safe orthographic & nukta groups
# Nukta insertion/deletion is only allowed for words where both forms are known
# to mean the exact same word (e.g. ज़रा/जरा, फ़ोन/फोन). Homographs like
# सजा (decorated) vs सज़ा (punishment) are excluded.
# =============================================================================
DEVANAGARI_GROUPS: List[Tuple[str, ...]] = [
    # Negation variants
    ("नहीं", "नही", "नहिं"),
    ("नाही", "नाहीं"),
    # Curated safe nukta pairs (meaning preserved identical)
    ("ज़रा", "जरा"),
    ("फ़ोन", "फोन"),
    ("ज़रूर", "जरूर"),
    ("ज़्यादा", "ज्यादा"),
    ("साफ़", "साफ"),
    ("फ़र्क", "फर्क"),
    ("क़रीब", "करीब"),
    ("फ़ायदा", "फायदा"),
    ("माफ़", "माफ"),
    ("तारीफ़", "तारीफ"),
    ("काग़ज़", "कागज"),
    ("बाज़ार", "बाजार"),
    ("नुक़सान", "नुकसान"),
    ("मज़बूत", "मजबूत"),
    ("हफ़्ता", "हफ्ता"),
    ("तरफ़", "तरफ"),
    ("क़ीमत", "कीमत"),
    ("क़ानून", "कानून"),
    ("फ़िल्म", "फिल्म"),
    ("ज़िंदगी", "जिंदगी"),
    ("ख़ास", "खास"),
    ("ख़बर", "खबर"),
    ("ख़ुश", "खुश"),
    ("ग़लत", "गलत"),
    ("ग़रीब", "गरीब"),
    ("वज़ह", "वजह"),
]

# Cross-script 1-to-1 high frequency mappings for opt-in cross_script perturbations
CROSS_SCRIPT_MAP: Dict[str, str] = {
    "nahi": "नहीं",
    "nhi": "नहीं",
    "nahin": "नहीं",
    "bahut": "बहुत",
    "bohot": "बहुत",
    "accha": "अच्छा",
    "achha": "अच्छा",
    "acha": "अच्छा",
    "theek": "ठीक",
    "thik": "ठीक",
    "shukriya": "शुक्रिया",
    "namaste": "नमस्ते",
    "dhanyavad": "धन्यवाद",
    "dhanyavaad": "धन्यवाद",
    "kya": "क्या",
    "kyun": "क्यों",
    "haan": "हाँ",
    "order": "ऑर्डर",
    "cancel": "कैंसिल",
    "delivery": "डिलीवरी",
    "refund": "रिफंड",
    "kharab": "खराब",
    "sahi": "सही",
    "milala": "मिळाला",
    "avadla": "आवडला",
    "pahije": "पाहिजे",
}
CROSS_SCRIPT_REV: Dict[str, str] = {
    "नहीं": "nahi",
    "बहुत": "bahut",
    "अच्छा": "accha",
    "ठीक": "theek",
    "शुक्रिया": "shukriya",
    "नमस्ते": "namaste",
    "धन्यवाद": "dhanyavad",
    "क्या": "kya",
    "क्यों": "kyun",
    "हाँ": "haan",
    "ऑर्डर": "order",
    "कैंसिल": "cancel",
    "डिलीवरी": "delivery",
    "रिफंड": "refund",
    "खराब": "kharab",
    "सही": "sahi",
    "मिळाला": "milala",
    "आवडला": "avadla",
    "पाहिजे": "pahije",
}

SAFE_GROUPS: List[Tuple[str, ...]] = _CORE + DEVANAGARI_GROUPS
CORE_GROUPS: List[Tuple[str, ...]] = SAFE_GROUPS  # public alias

# =============================================================================
# LEXICON — extra single words the normalizer de-stretches ("lateee" → "late")
# =============================================================================
LEXICON: List[str] = [
    # English / Hindi customer support
    "late", "order", "cancel", "wrong", "parcel", "courier", "package", "tracking",
    "upi", "card", "amount", "human", "agent", "manager", "representative", "senior",
    "item", "galat", "size", "paise", "delivered", "damaged", "double", "call", "care",
    "return", "complaint", "support", "payment", "refund", "product", "quality",
    # Marathi
    "milala", "mileli", "pathavla", "aala", "aali", "gela", "geli",
    "sangto", "sangti", "kara", "kela", "keli", "pahije", "nako",
    "ghya", "dya", "thamba", "avadla", "avadli",
    # Devanagari words
    "ऑर्डर", "सामान", "पैसा", "पैसे", "मदद", "शिकायत", "वापस", "खराब", "अच्छा", "उत्पाद", "पहुंचा",
    "डिलीवरी", "रिफंड", "कैंसिल",
    # Bengali
    "bhalo", "dhonnobad", "achi",
    # Tamil
    "vanakkam", "nandri", "romba", "seri", "illa",
    # Telugu
    "bagundi", "manchidi", "namaskaram", "kavali",
    # Kannada
    "chennagide", "bekku", "maadu", "namaskara",
    # Gujarati
    "majama", "dhanyavaad",
    # Punjabi
    "tussi", "kiddan", "shukriya",
    # Malayalam
    "njaan", "entha", "cheyyunnu",
]

# =============================================================================
# AMBIGUOUS — tokens we deliberately never rewrite
# =============================================================================
AMBIGUOUS: Dict[str, str] = {
    "nai":  "Hindi: 'no' (nahi) OR 'new' (nayi) — context-dependent",
    "he":   "Marathi 'this/he' vs Hindi 'is' (hai) — context-dependent",
    "to":   "Hindi connector/filler vs English preposition 'to' — ambiguous",
    "ho":   "Hindi 'you are' vs Marathi/Punjabi 'to be' — context-dependent",
    "re":   "Hindi/Marathi address suffix vs Bengali informal pronoun — context-dependent",
    "ki":   "Hindi 'of/that' vs Bengali/Assamese 'what' — context-dependent",
    "ko":   "Hindi postposition 'to' vs other uses",
    "po":   "Tamil/Kannada/Malayalam 'go (imperative)' — too short and ambiguous in mixed text",
    # Devanagari ambiguous tokens
    "ना":   "Hindi/Marathi: tag particle vs negation 'na' — ambiguous",
    "न":    "Hindi/Marathi: single-letter conjunction/filler vs negation 'na' — ambiguous",
}

# =============================================================================
# NEGATIONS — safe variants never add or remove these tokens
# =============================================================================
NEGATIONS: frozenset = frozenset({
    # Hindi / Urdu / Marathi (Roman)
    "nahi", "nhi", "nahin", "nahee", "nako", "nko", "mat", "maat",
    # Hindi / Marathi (Devanagari)
    "नहीं", "नही", "नहिं", "नाही", "नाहीं", "नको", "नकोस", "नका", "मत", "ना", "न",
    # English
    "not", "no", "never",
    # Dravidian
    "illa", "illai", "beda", "ledu", "vaddu", "kaadu", "venda", "alla",
    # Bengali
    "noy", "nei",
    # Gujarati
    "nathi",
})

# =============================================================================
# Low-level utilities
# =============================================================================
# Devanagari block U+0900–U+097F includes consonants, vowels, matras (combining
# vowel signs), viramas, anusvara, visarga etc.  Python's \w+ does NOT match
# combining marks (Unicode category Mn/Mc), so "ऑर्डर" splits at virama into
# ["ऑर","डर"] and "नहीं" loses its anusvara.  We add the full block first as an
# alternation so every Devanagari word is captured as one token (excluding
# danda \u0964 and double danda \u0965 which are punctuation).
_DEVA_WORD = r"[\u0900-\u0963\u0966-\u097F]+"
WORD_RE = re.compile(_DEVA_WORD + r"|\w+", re.UNICODE)
_SPLIT_RE = re.compile(_DEVA_WORD + r"|\w+|\W+", re.UNICODE)
_RUN_RE = re.compile(r"(.)\1+")


def tokenize_keep(text: str) -> List[str]:
    """Split into word and non-word chunks; ''.join(result) == text."""
    return _SPLIT_RE.findall(text)


def is_word(chunk: str) -> bool:
    return bool(chunk) and (chunk[0].isalnum() or chunk[0] == "_")


_DEVA_TO_ASCII_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def word_key(word: str) -> str:
    """Lowercase + NFC-normalize + digits to ASCII + collapse repeated letters: 'Naahi' -> 'nahi', 'avadlaaa' -> 'avadla'."""
    nfc = unicodedata.normalize("NFC", word)
    norm = nfc.translate(_DEVA_TO_ASCII_DIGITS)
    return _RUN_RE.sub(r"\1", norm.lower())


class RuleSet:
    """A reviewed set of language rules. Build one per project or language mix.

    Quick start::

        from fliprate.rules import DEFAULT_RULES, LANGUAGE_GROUPS

        # add Tamil on top of defaults
        rules = DEFAULT_RULES.extended(groups=LANGUAGE_GROUPS["tamil"])

        # Telugu + Kannada custom build
        from fliprate.rules import CORE_GROUPS, RuleSet
        rules = RuleSet(groups=CORE_GROUPS + LANGUAGE_GROUPS["telugu"] + LANGUAGE_GROUPS["kannada"])

        # All languages (includes MEDIUM/LOW)
        rules = RuleSet(groups=sum(LANGUAGE_GROUPS.values(), []))

    Fields
    ------
    groups     : spellings that are the same word (first member = canonical)
    lexicon    : extra words the normalizer may de-stretch ("lateee" → "late")
    negations  : tokens that reverse meaning; variants never add/remove them
    ambiguous  : tokens that must never be rewritten, with reason string
    """

    def __init__(self, groups=None, lexicon=None, negations=None, ambiguous=None):
        raw = list(SAFE_GROUPS if groups is None else groups)
        self.lexicon: List[str] = list(LEXICON if lexicon is None else lexicon)
        self.negations: frozenset = frozenset(NEGATIONS if negations is None else negations)
        self.ambiguous: Dict[str, str] = dict(AMBIGUOUS if ambiguous is None else ambiguous)
        self.groups: List[Tuple[str, ...]] = []
        self.key_to_group: Dict[str, int] = {}
        self.key_to_canon: Dict[str, str] = {}

        gi = 0
        for group in raw:
            group = tuple(m for m in group if m)
            if len(group) < 2:
                continue
            # Check for ambiguous members
            for member in group:
                if member.lower() in self.ambiguous:
                    raise ValueError(f"{member!r} is marked ambiguous and cannot be in a safe group")
            # Deduplicate by word_key within this group (keep first occurrence)
            seen_keys: dict = {}
            deduped = []
            for member in group:
                k = word_key(member)
                if k not in seen_keys:
                    seen_keys[k] = True
                    deduped.append(member)
            if len(deduped) < 2:
                continue
            group = tuple(deduped)
            # Check for cross-group key conflicts — skip conflicting groups rather than crash
            conflict = False
            for member in group:
                k = word_key(member)
                existing = self.key_to_group.get(k)
                if existing is not None and existing != gi:
                    conflict = True
                    break
            if conflict:
                continue
            # Register
            for member in group:
                k = word_key(member)
                self.key_to_group[k] = gi
                self.key_to_canon[k] = group[0]
            self.groups.append(group)
            gi += 1

        for word in self.lexicon:
            self.key_to_canon.setdefault(word_key(word), word)

    # ── lookups ───────────────────────────────────────────────────────────────
    def group_of(self, word: str) -> Optional[int]:
        return self.key_to_group.get(word_key(word))

    def canonical_tokens(self, text: str, cross_script: bool = False) -> Tuple[str, ...]:
        """Meaning skeleton: lowercase, repeats collapsed, safe spellings unified."""
        out = []
        for w in WORD_RE.findall(text):
            k = word_key(w)
            canon = self.key_to_canon.get(k, k)
            if cross_script:
                canon = CROSS_SCRIPT_MAP.get(canon, canon)
            out.append(canon)
        return tuple(out)

    def negation_count(self, text: str) -> int:
        count = 0
        for w in WORD_RE.findall(text):
            canon = self.key_to_canon.get(word_key(w))
            if w.lower() in self.negations or canon in self.negations:
                count += 1
        return count

    def is_safe_variant(self, original: str, variant: str, cross_script: bool = False) -> bool:
        """True if `variant` keeps the same words (up to safe spelling) and same negation count."""
        if not isinstance(original, str):
            raise TypeError(f"text must be str, got {type(original).__name__}")
        if not isinstance(variant, str):
            raise TypeError(f"text must be str, got {type(variant).__name__}")
        return (self.canonical_tokens(original, cross_script=cross_script) == self.canonical_tokens(variant, cross_script=cross_script)
                and self.negation_count(original) == self.negation_count(variant))

    def ambiguous_tokens(self, text: str) -> List[str]:
        return sorted({w.lower() for w in WORD_RE.findall(text) if w.lower() in self.ambiguous})

    # ── extension ─────────────────────────────────────────────────────────────
    def extended(self, groups: Iterable[Iterable[str]] = (), lexicon: Iterable[str] = (),
                 negations: Iterable[str] = (), ambiguous: Optional[Dict[str, str]] = None) -> "RuleSet":
        """Return a new RuleSet with extra reviewed rules appended.
        A group that would merge two existing distinct groups raises ValueError."""
        merged = [list(g) for g in self.groups]
        for raw in groups:
            new = list(dict.fromkeys(w.strip().lower() for w in raw if w.strip()))
            if len(new) < 2:
                continue
            index = {word_key(m): gi for gi, g in enumerate(merged) for m in g}
            hit = {index[word_key(w)] for w in new if word_key(w) in index}
            if len(hit) > 1:
                raise ValueError(f"Spellings {new} would merge two different existing groups")
            if hit:
                gi = hit.pop()
                have = {word_key(m) for m in merged[gi]}
                merged[gi].extend(w for w in new if word_key(w) not in have)
            else:
                merged.append(new)
        amb = dict(self.ambiguous)
        amb.update(ambiguous or {})
        return RuleSet(merged, list(self.lexicon) + list(lexicon),
                       self.negations | set(negations), amb)

    @classmethod
    def from_csv(cls, path, base: Optional["RuleSet"] = None) -> "RuleSet":
        """Load reviewed spelling groups from a CSV with a `spellings` column.
        Each row: pipe-separated spellings (e.g. "nahi|nhi|nahin").
        If a `status` column exists only rows with approve/approved/yes/y are used."""
        import csv
        groups = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "spellings" not in reader.fieldnames:
                raise ValueError("Rules CSV needs a `spellings` column (e.g. nahi|nhi|nahin)")
            has_status = "status" in reader.fieldnames
            for row in reader:
                if has_status and (row.get("status") or "").strip().lower() not in (
                        "approve", "approved", "yes", "y"):
                    continue
                groups.append((row["spellings"] or "").split("|"))
        return (base or DEFAULT_RULES).extended(groups=groups)

    def fingerprint(self) -> str:
        """Short hash stored in every report for reproducibility."""
        blob = json.dumps(
            {"g": sorted(sorted(g) for g in self.groups),
             "l": sorted(self.lexicon),
             "n": sorted(self.negations),
             "a": sorted(self.ambiguous)},
            sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# =============================================================================
# Module-level singletons
# =============================================================================
DEFAULT_RULES = RuleSet()

# Backwards-compatible aliases
GROUPS = DEFAULT_RULES.groups
KEY_TO_GROUP = DEFAULT_RULES.key_to_group
KEY_TO_CANON = DEFAULT_RULES.key_to_canon


def group_of(word: str) -> Optional[int]:
    return DEFAULT_RULES.group_of(word)


def canonical_tokens(text: str) -> Tuple[str, ...]:
    return DEFAULT_RULES.canonical_tokens(text)


def negation_count(text: str) -> int:
    return DEFAULT_RULES.negation_count(text)


def is_safe_variant(original: str, variant: str, cross_script: bool = False) -> bool:
    return DEFAULT_RULES.is_safe_variant(original, variant, cross_script=cross_script)


def ambiguous_tokens(text: str) -> List[str]:
    return DEFAULT_RULES.ambiguous_tokens(text)


_LANG_ALIAS_MAP = {
    "hi": "hindi", "hin": "hindi", "hindi": "hindi",
    "mr": "marathi", "mar": "marathi", "marathi": "marathi",
    "ta": "tamil", "tam": "tamil", "tamil": "tamil",
    "te": "telugu", "tel": "telugu", "telugu": "telugu",
    "kn": "kannada", "kan": "kannada", "kannada": "kannada",
    "ml": "malayalam", "mal": "malayalam", "malayalam": "malayalam",
    "gu": "gujarati", "guj": "gujarati", "gujarati": "gujarati",
    "pa": "punjabi", "pan": "punjabi", "punjabi": "punjabi",
    "bn": "bengali", "ben": "bengali", "bengali": "bengali",
    "ur": "urdu", "urd": "urdu", "urdu": "urdu",
    "or": "odia", "ori": "odia", "odia": "odia",
    "as": "assamese", "asm": "assamese", "assamese": "assamese",
}


def load_rules(language: str) -> RuleSet:
    """Load a RuleSet configured for a specific language by name or code.

    Examples:
        rules = fr.load_rules("hindi")
        rules = fr.load_rules("marathi")
        rules = fr.load_rules("ta")
    """
    lang = language.strip().lower()
    canonical_lang = _LANG_ALIAS_MAP.get(lang, lang)
    if canonical_lang in LANGUAGE_GROUPS:
        return DEFAULT_RULES.extended(groups=LANGUAGE_GROUPS[canonical_lang])
    raise ValueError(
        f"Unknown language '{language}'. Supported: {', '.join(sorted(set(_LANG_ALIAS_MAP.values())))}"
    )


def add_custom_rule(variants: Sequence[str], canonical: Optional[str] = None,
                    rules: Optional[RuleSet] = None) -> RuleSet:
    """Add a group of equivalent spellings to an existing or default RuleSet.

    Example:
        rules = fr.add_custom_rule(["discnt", "dscnt", "discount"], canonical="discount")
    """
    base_rules = rules or DEFAULT_RULES
    all_words = list(variants)
    if canonical and canonical not in all_words:
        all_words.append(canonical)
    words_unique = sorted(set(all_words))
    if canonical and canonical in words_unique:
        new_group = (canonical,) + tuple(w for w in words_unique if w != canonical)
    else:
        new_group = tuple(words_unique)
    return base_rules.extended(groups=[new_group])


def preview_variants(text: str, n: int = 10, **kwargs) -> List[str]:
    """Preview a list of generated variant strings for quick interactive testing."""
    from .perturbation import perturb
    return [v.text for v in perturb(text, n=n, **kwargs)]


def describe_rule_group(word_or_group: str, rules: Optional[RuleSet] = None) -> Optional[Dict[str, Any]]:
    """Describe the rule group containing a word, including variants and canonical spelling."""
    target_rules = rules or DEFAULT_RULES
    w_key = word_key(word_or_group)
    idx = target_rules.key_to_group.get(w_key)
    if idx is None:
        return None
    group = target_rules.groups[idx]
    canonical = target_rules.key_to_canon.get(w_key, group[0])
    is_neg = (w_key in target_rules.negations or word_or_group.lower() in target_rules.negations)
    return {
        "word": word_or_group,
        "canonical": canonical,
        "group_id": idx,
        "variants": list(group),
        "variant_count": len(group),
        "is_negation": is_neg,
    }

