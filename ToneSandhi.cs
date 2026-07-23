namespace MisakiSharp;

/// <summary> Mandarin tone sandhi rules, a faithful port of misaki's tone_sandhi.py (itself adapted from PaddleSpeech). </summary>
/// <remarks> Rewrites pypinyin-style finals ("ia1 i3") for neutral tones plus 不/一/third-tone shifts, and pre-merges jieba's segmentation quirks so the rules see whole words. Wire <see cref="CutForSearch"/> and <see cref="FinalsForWord"/> before use. </remarks>
public static class ToneSandhi {
    /// <summary> jieba.cut_for_search stand-in, returning tokens in jieba's emission order (we stable-sort by length ourselves). </summary>
    public static Func<string, List<string>> CutForSearch;

    /// <summary> lazy_pinyin(word, FINALS_TONE3, neutral_tone_with_five) stand-in. Must return a fresh list each call, since 嗯 entries get patched in place. </summary>
    public static Func<string, List<string>> FinalsForWord;

    const string SandhiPunctuation = "、：，；。？！“”‘’':,;.?!";

    static readonly HashSet<string> mustNeuralToneWords = [
        "麻烦", "麻利", "鸳鸯", "高粱", "骨头", "骆驼", "马虎", "首饰", "馒头", "馄饨", "风筝",
        "难为", "队伍", "阔气", "闺女", "门道", "锄头", "铺盖", "铃铛", "铁匠", "钥匙", "里脊",
        "里头", "部分", "那么", "道士", "造化", "迷糊", "连累", "这么", "这个", "运气", "过去",
        "软和", "转悠", "踏实", "跳蚤", "跟头", "趔趄", "财主", "豆腐", "讲究", "记性", "记号",
        "认识", "规矩", "见识", "裁缝", "补丁", "衣裳", "衣服", "衙门", "街坊", "行李", "行当",
        "蛤蟆", "蘑菇", "薄荷", "葫芦", "葡萄", "萝卜", "荸荠", "苗条", "苗头", "苍蝇", "芝麻",
        "舒服", "舒坦", "舌头", "自在", "膏药", "脾气", "脑袋", "脊梁", "能耐", "胳膊", "胭脂",
        "胡萝", "胡琴", "胡同", "聪明", "耽误", "耽搁", "耷拉", "耳朵", "老爷", "老实", "老婆",
        "戏弄", "将军", "翻腾", "罗嗦", "罐头", "编辑", "结实", "红火", "累赘", "糨糊", "糊涂",
        "精神", "粮食", "簸箕", "篱笆", "算计", "算盘", "答应", "笤帚", "笑语", "笑话", "窟窿",
        "窝囊", "窗户", "稳当", "稀罕", "称呼", "秧歌", "秀气", "秀才", "福气", "祖宗", "砚台",
        "码头", "石榴", "石头", "石匠", "知识", "眼睛", "眯缝", "眨巴", "眉毛", "相声", "盘算",
        "白净", "痢疾", "痛快", "疟疾", "疙瘩", "疏忽", "畜生", "生意", "甘蔗", "琵琶", "琢磨",
        "琉璃", "玻璃", "玫瑰", "玄乎", "狐狸", "状元", "特务", "牲口", "牙碜", "牌楼", "爽快",
        "爱人", "热闹", "烧饼", "烟筒", "烂糊", "点心", "炊帚", "灯笼", "火候", "漂亮", "滑溜",
        "溜达", "温和", "清楚", "消息", "浪头", "活泼", "比方", "正经", "欺负", "模糊", "槟榔",
        "棺材", "棒槌", "棉花", "核桃", "栅栏", "柴火", "架势", "枕头", "枇杷", "机灵", "本事",
        "木头", "木匠", "朋友", "月饼", "月亮", "暖和", "明白", "时候", "新鲜", "故事", "收拾",
        "收成", "提防", "挖苦", "挑剔", "指甲", "指头", "拾掇", "拳头", "拨弄", "招牌", "招呼",
        "抬举", "护士", "折腾", "扫帚", "打量", "打算", "打扮", "打听", "打发", "扎实", "扁担",
        "戒指", "懒得", "意识", "意思", "悟性", "怪物", "思量", "怎么", "念头", "念叨", "别人",
        "快活", "忙活", "志气", "心思", "得罪", "张罗", "弟兄", "开通", "应酬", "庄稼", "干事",
        "帮手", "帐篷", "希罕", "师父", "师傅", "巴结", "巴掌", "差事", "工夫", "岁数", "屁股",
        "尾巴", "少爷", "小气", "小伙", "将就", "对头", "对付", "寡妇", "家伙", "客气", "实在",
        "官司", "学问", "字号", "嫁妆", "媳妇", "媒人", "婆家", "娘家", "委屈", "姑娘", "姐夫",
        "妯娌", "妥当", "妖精", "奴才", "女婿", "头发", "太阳", "大爷", "大方", "大意", "大夫",
        "多少", "多么", "外甥", "壮实", "地道", "地方", "在乎", "困难", "嘴巴", "嘱咐", "嘟囔",
        "嘀咕", "喜欢", "喇嘛", "喇叭", "商量", "唾沫", "哑巴", "哈欠", "哆嗦", "咳嗽", "和尚",
        "告诉", "告示", "含糊", "吓唬", "后头", "名字", "名堂", "合同", "吆喝", "叫唤", "口袋",
        "厚道", "厉害", "千斤", "包袱", "包涵", "匀称", "勤快", "动静", "动弹", "功夫", "力气",
        "前头", "刺猬", "刺激", "别扭", "利落", "利索", "利害", "分析", "出息", "凑合", "凉快",
        "冷战", "冤枉", "冒失", "养活", "关系", "先生", "兄弟", "便宜", "使唤", "佩服", "作坊",
        "体面", "位置", "似的", "伙计", "休息", "什么", "人家", "亲戚", "亲家", "交情", "云彩",
        "事情", "买卖", "主意", "丫头", "丧气", "两口", "东西", "东家", "世故", "不由", "下水",
        "下巴", "上头", "上司", "丈夫", "丈人", "一辈", "那个", "菩萨", "父亲", "母亲", "咕噜",
        "邋遢", "费用", "冤家", "甜头", "介绍", "荒唐", "大人", "泥鳅", "幸福", "熟悉", "计划",
        "扑腾", "蜡烛", "姥爷", "照顾", "喉咙", "吉他", "弄堂", "蚂蚱", "凤凰", "拖沓", "寒碜",
        "糟蹋", "倒腾", "报复", "逻辑", "盘缠", "喽啰", "牢骚", "咖喱", "扫把", "惦记",
    ];

    static readonly HashSet<string> mustNotNeuralToneWords = [
        "男子", "女子", "分子", "原子", "量子", "莲子", "石子", "瓜子", "电子", "人人", "虎虎",
        "幺幺", "干嘛", "学子", "哈哈", "数数", "袅袅", "局地", "以下", "娃哈哈", "花花草草", "留得",
        "耕地", "想想", "熙熙", "攘攘", "卵子", "死死", "冉冉", "恳恳", "佼佼", "吵吵", "打打",
        "考考", "整整", "莘莘", "落地", "算子", "家家户户", "青青",
    ];

    /// <summary> Fixes jieba's segmentation before sandhi -- 不, 一, reduplications, third-tone runs and 儿 get glued back onto their neighbors. </summary>
    public static List<(string word, string pos)> PreMergeForModify(List<(string word, string pos)> seg) {
        seg = MergeYi(MergeBu(seg));
        seg = MergeThreeTones(MergeReduplication(seg), wholeWords: true);
        return MergeEr(MergeThreeTones(seg, wholeWords: false));
    }

    /// <summary> Applies all sandhi passes (不, 一, neutral, third-tone -- in that order) to one word's finals, returning the rewritten list. </summary>
    public static List<string> ModifiedTone(string word, string pos, List<string> finals) {
        finals = YiSandhi(word, BuSandhi(word, finals));
        return ThreeSandhi(word, NeuralSandhi(word, pos, finals));
    }

    /// <summary> 不 goes neutral mid-ABA ("看不懂"), and second tone before a fourth ("不怕"). </summary>
    static List<string> BuSandhi(string word, List<string> finals) {
        if (word.Length == 3 && word[1] == '不') { finals[1] = WithTone(finals[1], '5'); return finals; }
        for (int i = 0; i < word.Length; i++) {
            if (word[i] == '不' && i + 1 < word.Length && finals[i + 1][^1] == '4') { finals[i] = WithTone(finals[i], '2'); }
        }
        return finals;
    }

    /// <summary> 一 keeps its tone in digit runs, is neutral between reduplications ("看一看"), first in 第一, else second before tone 4/5 and fourth otherwise (unless punctuation follows). </summary>
    static List<string> YiSandhi(string word, List<string> finals) {
        if (word.Contains('一') && word.All(c => c == '一' || IsPythonNumeric(c))) { return finals; }
        if (word.Length == 3 && word[1] == '一' && word[0] == word[^1]) { finals[1] = WithTone(finals[1], '5'); }
        else if (word.StartsWith("第一")) { finals[1] = WithTone(finals[1], '1'); }
        else {
            for (int i = 0; i < word.Length - 1; i++) {
                if (word[i] != '一') { continue; }
                if (finals[i + 1][^1] is '4' or '5') { finals[i] = WithTone(finals[i], '2'); }
                else if (!SandhiPunctuation.Contains(word[i + 1])) { finals[i] = WithTone(finals[i], '4'); }
            }
        }
        return finals;
    }

    /// <summary> Neutral-tone pass -- reduplications, particle/suffix tails (的/了/们/上/来...), measure-word 个, and the big known-word list. </summary>
    /// <remarks> After the rule chain, the word also gets split in two (via <see cref="SplitWord"/>) and each half is checked against the known-word list again. </remarks>
    static List<string> NeuralSandhi(string word, string pos, List<string> finals) {
        if (mustNotNeuralToneWords.Contains(word)) { return finals; }
        for (int j = 1; j < word.Length; j++) {
            if (word[j] == word[j - 1] && pos[0] is 'n' or 'v' or 'a') { finals[j] = WithTone(finals[j], '5'); }
        }
        int geIdx = word.IndexOf('个');
        if (word.Length >= 1 && "吧呢啊呐噻嘛吖嗨呐哦哒滴哩哟喽啰耶喔诶".Contains(word[^1])) { finals[^1] = WithTone(finals[^1], '5'); }
        else if (word.Length >= 1 && "的地得".Contains(word[^1])) { finals[^1] = WithTone(finals[^1], '5'); }
        else if (word.Length == 1 && "了着过".Contains(word) && pos is "ul" or "uz" or "ug") { finals[^1] = WithTone(finals[^1], '5'); }
        else if (word.Length > 1 && "们子".Contains(word[^1]) && pos is "r" or "n") { finals[^1] = WithTone(finals[^1], '5'); }
        else if (word.Length > 1 && "上下".Contains(word[^1]) && pos is "s" or "l" or "f") { finals[^1] = WithTone(finals[^1], '5'); }
        else if (word.Length > 1 && "来去".Contains(word[^1]) && "上下进出回过起开".Contains(word[^2])) { finals[^1] = WithTone(finals[^1], '5'); }
        else if ((geIdx >= 1 && (IsPythonNumeric(word[geIdx - 1]) || "几有两半多各整每做是".Contains(word[geIdx - 1]))) || word == "个") { finals[geIdx] = WithTone(finals[geIdx], '5'); }
        else if (mustNeuralToneWords.Contains(word) || mustNeuralToneWords.Contains(LastTwo(word))) { finals[^1] = WithTone(finals[^1], '5'); }

        var wordList = SplitWord(word);
        int headLength = Math.Min(wordList[0].Length, finals.Count);
        var finalsList = new[] { finals.GetRange(0, headLength), finals.GetRange(headLength, finals.Count - headLength) };
        for (int i = 0; i < 2; i++) {
            if (mustNeuralToneWords.Contains(wordList[i]) || mustNeuralToneWords.Contains(LastTwo(wordList[i]))) { finalsList[i][^1] = WithTone(finalsList[i][^1], '5'); }
        }
        return [.. finalsList[0], .. finalsList[1]];
    }

    /// <summary> Third-tone chains -- in a run of third tones every syllable but the last turns second, split-point-aware for 3-char words ("蒙古/包" vs "纸/老虎") and pairwise for 4-char idioms. </summary>
    static List<string> ThreeSandhi(string word, List<string> finals) {
        if (word.Length == 2 && AllToneThree(finals)) { finals[0] = WithTone(finals[0], '2'); }
        else if (word.Length == 3) {
            var wordList = SplitWord(word);
            if (AllToneThree(finals)) {
                if (wordList[0].Length == 2) { finals[0] = WithTone(finals[0], '2'); finals[1] = WithTone(finals[1], '2'); }
                else if (wordList[0].Length == 1) { finals[1] = WithTone(finals[1], '2'); }
            } else {
                int headLength = Math.Min(wordList[0].Length, finals.Count);
                var finalsList = new[] { finals.GetRange(0, headLength), finals.GetRange(headLength, finals.Count - headLength) };
                for (int i = 0; i < 2; i++) {
                    var sub = finalsList[i];
                    if (AllToneThree(sub) && sub.Count == 2) { sub[0] = WithTone(sub[0], '2'); }
                    else if (i == 1 && !AllToneThree(sub) && sub[0][^1] == '3' && finalsList[0][^1][^1] == '3') { finalsList[0][^1] = WithTone(finalsList[0][^1], '2'); }
                }
                finals = [.. finalsList[0], .. finalsList[1]];
            }
        } else if (word.Length == 4) {
            int headLength = Math.Min(2, finals.Count);
            var finalsList = new[] { finals.GetRange(0, headLength), finals.GetRange(headLength, finals.Count - headLength) };
            finals = [];
            foreach (var sub in finalsList) {
                if (AllToneThree(sub)) { sub[0] = WithTone(sub[0], '2'); }
                finals.AddRange(sub);
            }
        }
        return finals;
    }

    /// <summary> Splits a word in two the way the original does -- shortest cut_for_search token plus the remainder, keeping original order. </summary>
    static List<string> SplitWord(string word) {
        var shortestToken = CutForSearch(word).OrderBy(token => token.Length).First();
        if (word.IndexOf(shortestToken, StringComparison.Ordinal) == 0) { return [shortestToken, word[shortestToken.Length..]]; }
        return [word[..^shortestToken.Length], shortestToken];
    }

    /// <summary> Glues a lone 不 onto the word after it, so 不-sandhi can see the pair. </summary>
    static List<(string word, string pos)> MergeBu(List<(string word, string pos)> seg) {
        List<(string word, string pos)> merged = [];
        for (int i = 0; i < seg.Count; i++) {
            var (word, pos) = seg[i];
            if (!IsXEng(pos) && i > 0 && seg[i - 1].word == "不") { word = "不" + word; }
            bool loneBu = word == "不" && i + 1 < seg.Count && !IsXEng(seg[i + 1].pos);
            if (!loneBu) { merged.Add((word, pos)); }
        }
        return merged;
    }

    /// <summary> Glues V-一-V reduplications ("听","一","听" -> "听一听") back together, then any lone 一 onto the word after it. </summary>
    static List<(string word, string pos)> MergeYi(List<(string word, string pos)> seg) {
        List<(string word, string pos)> merged = [];
        for (int i = 0; i < seg.Count; i++) {
            var (word, pos) = seg[i];
            if (i >= 1 && word == "一" && i + 1 < seg.Count && seg[i - 1].word == seg[i + 1].word && seg[i - 1].pos == "v" && !IsXEng(seg[i + 1].pos)) {
                merged[^1] = (merged[^1].word + "一" + seg[i + 1].word, merged[^1].pos);
                i++;
            } else { merged.Add((word, pos)); }
        }
        seg = merged;
        merged = [];
        foreach (var (word, pos) in seg) {
            if (merged.Count > 0 && merged[^1].word == "一" && !IsXEng(pos)) { merged[^1] = (merged[^1].word + word, merged[^1].pos); }
            else { merged.Add((word, pos)); }
        }
        return merged;
    }

    /// <summary> Glues immediately repeated words ("看","看" -> "看看") so neutral-tone reduplication rules can fire. </summary>
    static List<(string word, string pos)> MergeReduplication(List<(string word, string pos)> seg) {
        List<(string word, string pos)> merged = [];
        foreach (var (word, pos) in seg) {
            if (merged.Count > 0 && word == merged[^1].word && !IsXEng(pos)) { merged[^1] = (merged[^1].word + word, merged[^1].pos); }
            else { merged.Add((word, pos)); }
        }
        return merged;
    }

    /// <summary> Glues neighbors whose tones chain as thirds -- whole-word-all-third mode, or boundary-syllables-only mode -- unless the left one is a reduplication or the pair exceeds 3 chars. </summary>
    static List<(string word, string pos)> MergeThreeTones(List<(string word, string pos)> seg, bool wholeWords) {
        List<List<string>> subFinals = [];
        foreach (var (word, pos) in seg) {
            if (IsXEng(pos)) { subFinals.Add(["0"]); continue; }
            var finals = FinalsForWord(word);
            for (int i = 0; i < word.Length; i++) { if (word[i] == '嗯') { finals[i] = "n2"; } }
            subFinals.Add(finals);
        }
        var mergedInto = new bool[seg.Count];
        List<(string word, string pos)> merged = [];
        for (int i = 0; i < seg.Count; i++) {
            var (word, pos) = seg[i];
            bool tonesChain = !IsXEng(pos) && i >= 1 && !mergedInto[i - 1] && (wholeWords
                ? AllToneThree(subFinals[i - 1]) && AllToneThree(subFinals[i])
                : subFinals[i - 1][^1][^1] == '3' && subFinals[i][0][^1] == '3');
            if (tonesChain && !IsReduplication(seg[i - 1].word) && seg[i - 1].word.Length + word.Length <= 3) {
                merged[^1] = (merged[^1].word + word, merged[^1].pos);
                mergedInto[i] = true;
            } else { merged.Add((word, pos)); }
        }
        return merged;
    }

    /// <summary> Glues erhua 儿 onto the word before it. </summary>
    static List<(string word, string pos)> MergeEr(List<(string word, string pos)> seg) {
        List<(string word, string pos)> merged = [];
        for (int i = 0; i < seg.Count; i++) {
            var (word, pos) = seg[i];
            if (i >= 1 && word == "儿" && !IsXEng(merged[^1].pos)) { merged[^1] = (merged[^1].word + word, merged[^1].pos); }
            else { merged.Add((word, pos)); }
        }
        return merged;
    }

    static string WithTone(string final, char tone) => final[..^1] + tone;
    static string LastTwo(string word) => word.Length > 2 ? word[^2..] : word;
    static bool AllToneThree(List<string> finals) => finals.All(final => final[^1] == '3');
    static bool IsReduplication(string word) => word.Length == 2 && word[0] == word[1];
    static bool IsXEng(string pos) => pos is "x" or "eng";
    /// <summary> Mirrors python's str.isnumeric (verified char-exact over the whole BMP) -- .NET misses the hanzi numerals, since they're letters with Unicode numeric values. </summary>
    /// <remarks> The hex tail is CJK compatibility ideographs (invisible twins of 參拾兩零六陸什), built from codepoints so nothing can silently normalize them. </remarks>
    static bool IsPythonNumeric(char c) => char.IsNumber(c) || hanNumerals.Contains(c);
    static readonly string hanNumerals = "〇一二三四五六七八九十百千万亿兆零壹贰叁肆伍陆柒捌玖拾佰仟廿卅卌㐅㒃㠪㭍亖什仨億兩卄参參叄壱幺廾弌弍弎弐漆萬貮貳阡陌陸" + string.Concat(new[] { 0xF96B, 0xF973, 0xF978, 0xF9B2, 0xF9D1, 0xF9D3, 0xF9FD }.Select(char.ConvertFromUtf32));
}
