# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>

import collections
import mylib.txt_lib as lt

DpcKind = collections.namedtuple(typename='DpcKind',
                                 field_names=['Id', 'Name', 'Title', 'Aliases', 'Desc', 'Example'])


class DpcKinds:
    UNK = 0
    HED = 1
    PU = 2
    SBV = 3
    VOB = 4
    ATT = 5
    QUN = 6
    COO = 7
    APP = 8
    ADJ = 9
    POB = 10
    SIM = 11
    TMP = 12
    LOC = 13
    DE = 14
    DI = 15
    DEI = 16
    SUO = 17
    BA = 18
    BEI = 19
    ADV = 20
    CMP = 21
    DBL = 22
    CNJ = 23
    CS = 24
    MT = 25
    VV = 26
    FOB = 27
    DOB = 28
    TOP = 29
    IS = 30
    IC = 31
    DC = 32
    VNV = 33
    YGC = 34
    Count = 35
    ERR = Count

    @classmethod
    def parse(cls, kind_str):
        kind = cls._KindSearch.get(kind_str, cls.ERR)
        if kind == cls.ERR:
            kind_str = kind_str.upper().strip()
            kind_str = lt.clean_punc_or_symbol(kind_str, " -")
            kind = cls._KindSearch.get(kind_str, cls.ERR)
        return kind

    @classmethod
    def get(cls, kind):
        if isinstance(kind, str):
            kind = cls.parse(kind)
        if 0 <= kind <= cls.YGC:
            return cls.KindObjects[kind]
        return cls.KindObjects[-1]

    _KindSearch = {}

    @classmethod
    def build_search(cls):
        def add_search(key, value, unique=False):
            key = key.upper()
            assert not unique or key not in cls._KindSearch
            if unique:
                assert key not in cls._KindSearch
                cls._KindSearch[key] = value
            elif key not in cls._KindSearch:
                cls._KindSearch[key] = value
        for kind_obj in cls.KindObjects:
            add_search(kind_obj.Name, kind_obj.Id, True)
            add_search(kind_obj.Title, kind_obj.Id, True)
        for kind_obj in cls.KindObjects:
            if kind_obj.Title.endswith("关系") or kind_obj.Title.endswith("结构"):
                add_search(kind_obj.Title[:-2], kind_obj.Id)
            for alias in kind_obj.Aliases.split(';'):
                add_search(alias, kind_obj.Id)
                if alias.endswith("关系") or alias.endswith("结构"):
                    add_search(alias[:-2], kind_obj.Id)

    KindObjects = tuple([
        DpcKind(UNK, "UNK", "未知", "dep;rcmod;xsubj;unknown;待定", "", ""),  #
        DpcKind(HED, "HED", "核心", "head;ROOT;EOS;erased",
                "该核心是指整个句子的核心，一般是句子的核心词和虚拟词（<EOS>或ROOT）的依存关系。",
                "如：这/r 就是/v恩施/ns最/d]便宜/a的/u出租车/n（就是/v ← <EOS>/<EOS>）"),
        DpcKind(PU, "WP", "标点", "punct;PU;标点符号",
                "大部分标点依存于其前面句子的核心词上",
                "如：嗨→！"),
        DpcKind(SBV, "SBV", "主谓", "nsubj;top;cop;nsubjpass;subject-verb",  # 施事;关系主体;经验者
                "主谓关系是指名词和动作之间的关系。",
                "如：父亲/n 逝世/v １０/m 周年/q 之际/nd（父亲/n ← 逝世/v）。"),
        DpcKind(VOB, "VOB", "动宾", "iobj;dobj;attr;verb-object",  # 内容;类指;结果事件;终处所
                "对于动词和宾语之间的关系我们定义了两个层次，一是句子的谓语动词及其宾语之间的关系，我们定为OBJ，在下面的单句依存关系中说明；"
                "二是非谓语动词及其宾语的关系，即VOB。这两种关系在结构上没有区别，只是在语法功能上，OBJ中的两个词充当句子的谓语动词和宾语，VOB中的两个词构成动宾短语，作为句子的其他修饰成分。",
                "如：历时/v 三/m 天/q 三/m夜/q（历时/v → 天/q）。"),
        DpcKind(ATT, "ATT", "定中", "nn;amod;assm;clf;det;限定;定语;",  # 描述
                "定中关系就是定语和中心词之间的关系，定语对中心词起修饰或限制作用",
                "工人/n师傅/n（工人/n ← 师傅/n）"),
        DpcKind(ADV, "ADV", "状中", "advmod;neg;adverbial",  # 评论;程度
                "状中结构是谓词性的中心词和其前面的修饰语之间的关系，中心词做谓语时，前面的修饰成分即为句子的状语",
                "如：连夜/d 安排/v 就位/v（连夜/d ← 安排/v）。"),
        DpcKind(CMP, "CMP", "动补结构", "vmod;rcomp;complement",  # 时态依存;方位词依存;描写体;趋向动词依存
                "补语用于对核心动词的补充说明。",
                "如：做完了作业（做/v → 完）。"),
        DpcKind(QUN, "QUN", "数量", "range;nummod;ordmod;quantity",
                "数量关系是指量词或名词同前面的数词之间的关系，该关系中，数词作修饰成分，依存于量词或名词。",
                "如：三/m天/q（三/m ← 天/q）。"),
        DpcKind(COO, "COO", "并列", "conj;comod;coordinate",  # 连接依存
                "并列关系是指两个相同类型的词并列在一起。",
                "如：奔腾/v咆哮/v的怒江激流（奔腾/v → 咆哮/v）。"),
        DpcKind(APP, "APP", "同位", "appos;appositive",  # 同位语
                "同位语是指所指相同、句法功能也相同的两个并列的词或词组。",
                "如：我们大家 （我们 → 大家）。"),
        DpcKind(ADJ, "ADJ", "附加", "etc;plmod;adjunct",
                "附加关系是一些附属词语对名词等成分的一种补充说明，使意思更加完整，有时候去掉也不影响意思。",
                "如：约/d 二十/m 多/m 米/q 远/a 处/n （二十/m → 多/m，米/q → 远/a）。"),
        DpcKind(POB, "POB", "介宾", "pobj;preposition-object",  # 介词依存
                "介词和宾语之间的关系，介词的属性同动词相似。",
                "如：距/p球门/n（距/p → 球门/n）。"),
        DpcKind(TMP, "TMP", "时间", "tmod;temporal",
                "时间关系定义的是时间状语和其所修饰的中心动词之间的关系。",
                "如：十点以前到公司（以前 ← 到）。"),
        DpcKind(LOC, "LOC", "处所", "loc;locative",
                "处所关系定义的是处所状语和其所修饰的中心动词之间的关系，",
                "如：在公园里玩耍（在 ← 玩耍）。"),
        DpcKind(SIM, "SIM", "比拟", "prnmod;similarity",
                "比拟关系是汉语中用于表达比喻的一种修辞结构。",
                "如：炮筒/n 似的/u 望远镜/n（炮筒/n ← 似的/u）。"),
        DpcKind(DE, "DE", "的字", "assmod;cpm",
                "“的”字结构是指结构助词“的”和其前面的修饰语以及后面的中心词之间的关系。",
                "如：上海/ns 的/u 工人/n（上海/ns ← 的/u，的/u ← 工人/n）。"),
        DpcKind(DI, "DI", "地字", "dvpm",
                "“地”字结构在构成上同DE类似，只是在功能上不同，DI通常作状语修饰动词。",
                "如： 方便/a 地/u 告诉/v 计算机/n（方便/a ← 地/u，地/u ← 告诉/v）。"),
        DpcKind(DEI, "DEI", "得字", "",
                "助词“得”同其后的形容词或动词短语等构成“得”字结构，对前面的动词进行补充说明。",
                "如：讲/v 得/u 很/d 对/a（讲/v → 得/u，得/u → 对/a）。"),
        DpcKind(SUO, "SUO", "所字", "prtmod",
                "“所”字为一结构助词，后接一宾语悬空的动词做“的”字结构的修饰语，“的”字经常被省略，使结构更加简洁。",
                "如：机电/b 产品/n 所/u 占/v 比重/n 稳步/d 上升/v（所/u ← 占/v）。"),
        DpcKind(BA, "BA", "把字", "ba",
                "把字句是主谓句的一种，句中谓语一般都是及物动词。",
                "如：我们把豹子打死了（把/p → 豹子/n）。"),
        DpcKind(BEI, "BEI", "被字", "pass",
                "被字句是被动句，是主语接受动作的句子。",
                "如：豹子被我们打死了（豹子/n ← 被/p）。"),
        DpcKind(DBL, "DBL", "兼语", "mmod;double",
                "兼语句一般有两个动词，第二个动词是第一个动作所要表达的目的或产生的结果。",
                "如：（使/v → 人/n ，使/v → 惊叹/v）。"),
        DpcKind(CNJ, "CNJ", "关联词", "cc;prep;conjunction",
                "关联词语是复句的有机部分。",
                "如：只要他请客，我就来。（只要 ← 请 ，就 ← 来）。"),
        DpcKind(CS, "CS", "关联", "ccomp;conjunctive",
                "当句子中存在关联结构时，关联词所在的两个句子（或者两个部分）之间通过各部分的核心词发生依存关系CS。",
                "如：只要他请客，我就来。（请 ← 来）。"),
        DpcKind(MT, "MT", "语态", "asp;mood-tense",
                "汉语中，经常用一些助词表达句子的时态和语气，这些助词分语气助词，如：吧，啊，呢等；还有时态助词，如：着，了，过。",
                "如： [12]答应/v [13]孩子/n [14]们/k [15]的/u [16]要求/n [17]吧/u [18]（[12]答应/v ← [17]吧/u）。"),
        DpcKind(VV, "VV", "连谓", "dvpmod;verb-verb",
                "连谓结构是同多项谓词性成分连用、这些成分间没有语音停顿、书面标点，也没有关联词语，没有分句间的逻辑关系，且共用一个主语。",
                "如：美国总统来华访问。（来华/v → 访问/v）。"),
        DpcKind(FOB, "FOB", "前置宾语", "lobj;fronting object",  # 受事
                "在汉语中，有时将句子的宾语前置，或移置句首，或移置主语和谓语之间，以起强调作用，我认识这个人 ← 这个人我认识。",
                "如：他什么书都读（书/n ← 读/v）。"),
        DpcKind(DOB, "DOB", "双宾语", "double object",
                "动词后出现两个宾语的句子叫双宾语句，分别是直接宾语和间接宾语。",
                "如：我送她一束花。（送/v → 她/r，送/v → 花/n）。"),
        DpcKind(TOP, "TOP", "主题", "topic;主题前置",
                "在表达中，我们经常会先提出一个主题性的内容，然后对其进行阐述说明；而主题部分与后面的说明部分并没有直接的语法关系，主题部分依存于后面的核心成分。",
                "如：西直门，怎么走？（西直门 ← 走）。"),
        DpcKind(IS, "IS", "独立结构", "independent structure",
                "独立成分在句子中不与其他成分产生结构关系，但意义上又是全句所必需的，具有相对独立性的一种成分。",
                "如：事情明摆着，我们能不管吗？"),
        DpcKind(IC, "IC", "独立分句", "independent clause",
                "两个单句在结构上彼此独立，都有各自的主语和谓语。",
                "如：我是中国人，我们爱自己的祖国。（是 → 爱）"),
        DpcKind(DC, "DC", "依存分句", "pccomp;lccomp;rccomp;dependent clause",
                "两个单句在结构上不是各自独立的，后一个分句的主语在形式上被省略，但不是前一个分句的主语，而是存在于前一个分句的其他成分中，如宾语、主题等成分。"
                "规定后一个分句的核心词依存于前一个分句的核心词。该关系同连谓结构的区别是两个谓词是否为同一主语，如为同一主语，则为VV，否则为DC。",
                "如：大家/r叫/v 它/r “/wp 麻木/a 车/n ”/wp ，/wp 听/v起来/v 怪怪的/a 。/wp（叫/v → 听/v）。"),
        DpcKind(VNV, "VNV", "叠词关系", "verb-no-verb;verb-one-verb",
                "如果叠词被分开了，如“是 不 是”、“看一看”，那么这几个词先合并在一起，然后预存到其他词上，叠词的内部关系定义为：(是1→不；不→是2） 。",
                "如：(是1→不；不→是2）"),
        DpcKind(YGC, "YGC", "词组内", "CZ;WORD;一个词",
                "当专名或者联绵词等切散后，他们之间本身没有语法关系，应该合起来才是一个词。",
                "如：百→度。"),
        DpcKind(ERR, "ERR", "错误", "ERROR",
                "（关联错误，内部使用）",
                "（关联错误，内部使用）"),
    ])


DpcKinds.build_search()


def parse_tagged_text(tagged_dependency_text, concat_by='\u00a0'):
    head_t = ["HED", "HED", 0, DpcKinds.UNK]
    order = 0
    head = list(head_t)
    sentence = [head]

    sentence_list = []
    for item in tagged_dependency_text.split(concat_by):
        order += 1
        word, tags = item.split(lt.ColumnSeparator)
        if len(word) == 0:
            word = " "
        sentence_ends = tags.endswith('〗〗')
        tags = tags.replace('〖', '-').replace('〗', '')
        tags = tags.split('-')
        item = [word, tags[0], int(tags[3]), DpcKinds.parse(tags[2])]
        assert int(tags[1]) == order
        if int(tags[3]) == 0:
            head[2] = order
        sentence.append(item)
        if sentence_ends:
            sentence_list.append(sentence)
            order = 0
            head = list(head_t)
            sentence = [head]

    return sentence_list


class WordNode:

    def __init__(self, word, tag, index):
        self.word = word
        self.tag = tag
        self.index = index
        self.links = []


class DependencyTree:

    def __init__(self, word_dpc_list):
        parent_idx = word_dpc_list[0][2] - 1
        word_dpc_list = word_dpc_list[1:]
        self.word_nodes = []
        for idx, (word, tag, _, _) in enumerate(word_dpc_list):
            self.word_nodes.append(DependencyTree.Node(word, tag, idx))
        self.root_node = self.word_nodes[parent_idx]

        done_indices = set()
        todo_indices = [parent_idx]
        while todo_indices:
            p_idx = todo_indices.pop(0)
            done_indices.add(p_idx)
            p_node = self.word_nodes[p_idx]
            for idx, (word, _, to_idx, link_kind) in enumerate(word_dpc_list):
                to_idx -= 1
                if to_idx == p_idx and idx not in done_indices:
                    link = DependencyTree.Link(p_node, self.word_nodes[idx], link_kind, False)
                    p_node.links.append(link)
                    todo_indices.append(idx)
                elif to_idx >= 0 and to_idx not in done_indices and idx == p_idx:
                    link = DependencyTree.Link(p_node, self.word_nodes[to_idx], link_kind)
                    p_node.links.append(link)
                    todo_indices.append(to_idx)

    class Node:

        def __init__(self, word, tag, index):
            self.word = word
            self.tag = tag
            self.index = index
            self.links = []

    class Link:

        def __init__(self, parent, child, kind, parent_to_child=True):
            self.parent = parent
            self.child = child
            self.kind = kind
            self.parent_to_child = parent_to_child

    SkipKinds = {DpcKinds.UNK, DpcKinds.HED, DpcKinds.PU, DpcKinds.YGC, DpcKinds.ERR,
                 DpcKinds.TMP, DpcKinds.CNJ, DpcKinds.CS, DpcKinds.MT, DpcKinds.IS, DpcKinds.IC, DpcKinds.DC}

    def _find(self, node, node_filter, link_filter):
        if node_filter is None or node_filter(node):
            for link in node.links:
                if link_filter is None and link.kind not in self.SkipKinds or link_filter(link, link.child):
                    yield node, link, link.child
        for link in node.links:
            if node_filter is None or node_filter(link.child):
                if link_filter is None and link.kind not in self.SkipKinds or link_filter(link, node):
                    yield link.child, link, node
        for link in node.links:
            self._find(link.child, node_filter, link_filter)

    def find(self, node_tags, link_kinds, node_words=None, target_tag=None, target_words=None):
        if isinstance(node_tags, str):
            node_tags = [node_tags]
        if isinstance(link_kinds, str):
            link_kinds = [link_kinds]
        if isinstance(target_words, str):
            target_words = [target_words]
        if link_kinds:
            link_kinds = [DpcKinds.get(kind).Id for kind in link_kinds]

        def node_filter(node):
            if node_tags and not any(node.tag.startswith(tag) for tag in node_tags):
                return False
            if node_words and node.word not in node_words:
                return False
            return True

        def link_filter(link, target):
            # if target.word == "发布" and link.kind == 3:
            #     print(target.word, link.kind)
            if link_kinds and link.kind not in link_kinds:
                return False
            if link_kinds is None and link.kind in self.SkipKinds:
                return False
            if target_tag and not target.tag.startswith(target_tag):
                return False
            if target_words and target.word not in target_words:
                return False
            return True
        return self._find(self.root_node, node_filter, link_filter)


def find_test(file_name, word_tags, concat_by='\u00a0'):
    pattern_counter_search = collections.defaultdict(collections.Counter)
    for line in lt.load_lines(file_name, True, True):
        sentence_list = parse_tagged_text(line, concat_by)
        for sentence in sentence_list:
            dpc_tree = DependencyTree(sentence)
            for node, link, target in dpc_tree.find(word_tags, None):
                pattern_counter = pattern_counter_search[node.tag]
                direction = ">>" if node.index < target.index else "<<"
                pattern_counter[f"{direction}\t{DpcKinds.get(link.kind).Title}\t{target.word}"] += 1
    for cur_tag, pattern_counter in pattern_counter_search.items():
        pattern_count_list = sorted(pattern_counter.items(), key=lambda pair: pair[1], reverse=True)
        for pattern, count in pattern_count_list:
            if count < 10:
                break
            for comp_tag, comp_counter in pattern_counter_search.items():
                cc = comp_counter[pattern]
                if cc >= count // 5:
                    continue
                ratio = 1000 if cc <= 0 else count // cc
                print(f"{cur_tag}\t{pattern}\t{ratio}\t{count}\t{comp_tag}\t{cc}")


def find_show(file_name, node_tags, link_kinds, target_words, concat_by='\u00a0'):
    for line in lt.load_lines(file_name, True, True):
        sentence_list = parse_tagged_text(line, concat_by)
        for sentence in sentence_list:
            dpc_tree = DependencyTree(sentence)
            for node, link, target in dpc_tree.find(node_tags, link_kinds, target_words=target_words):
                direction = ">>" if node.index < target.index else "<<"
                print(f"{node.word}\t{direction}\t{DpcKinds.get(link.kind).Title}\t{target.word}")
                # content = sentence[node.index+1:target.index+2] \
                #     if node.index < target.index else \
                #     sentence[target.index+1:node.index+2]
                print("".join(wi[0] for wi in sentence[1:]))


if __name__ == '__main__':
    # find_test(r"E:\产销链图谱标注\sgb_train.dp.txt", ['nt', 'np', 'nc'])
    find_show(r"E:\产销链图谱标注\sgb_train.dp.txt", 'np', '主谓', '布局')
