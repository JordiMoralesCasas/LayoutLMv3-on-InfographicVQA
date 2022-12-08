from preprocess.utils import clean_text, get_answer_indices, better_subfinder, get_answer_indices_by_enumeration
import editdistance

def extract_start_end_index_v1(current_answers, words):
    """
    Adapted from https://github.com/anisha2102/docvqa/blob/master/create_dataset.py
    :param current_answers: List of answers
    :param words: List of all the words in the document
    :return:
    """
    ## extracting the start, end index
    processed_answers = []
    ## remove duplicates because of the case of multiple answers
    current_answers = list(set(current_answers))
    all_not_found = True
    for ans_index, current_ans in enumerate(current_answers):
        start_index, end_index, extracted_answer = get_answer_indices(words, current_ans)
        ans = current_ans.lower()
        extracted_answer = clean_text(extracted_answer)
        ans = clean_text(ans)
        dist = editdistance.eval(extracted_answer.replace(' ', ''),
                                 ans.replace(' ', '')) if extracted_answer != None else 1000
        if start_index == -1:
            end_index = -1
            extracted_answer = ""
        if dist > 5:
            start_index = -1
            end_index = -1
        if start_index == -1 or len(extracted_answer) > 150 or extracted_answer == "":
            start_index = -1
            end_index = -1
            extracted_answer = ""
        if start_index != -1:
            all_not_found = False
        processed_answers.append({
            "start_word_position": start_index,
            "end_word_position": end_index,
            "gold_answer": current_ans,
            "extracted_answer": extracted_answer})
    return processed_answers, all_not_found

def extract_start_end_index_v2(current_answers, words):
    """
    Follwing https://github.com/redthing1/layoutlm_experiments/blob/main/llm_tests/llm_tests/prep_docvqa_xqa.py.
    :param current_answers:
    :param words:
    :return:
    """
    ## extracting the start, end index
    processed_answers = []
    ## remove duplicates because of the case of multiple answers
    current_answers = list(set(current_answers))
    all_not_found = True
    for ans_index in range(len(current_answers)):
        current_ans = current_answers[ans_index]
        match, word_idx_start, word_idx_end = better_subfinder(
            words, current_ans.lower()
        )
        if not match:
            for i in range(len(current_ans)):
                if len(current_ans) == 1:
                    # this method won't work for single-character answers
                    break  # break inner loop
                # drop the ith character from the answer
                answer_i = current_ans[:i] + current_ans[i + 1:]
                # print('Trying: ', i, answer, answer_i, answer_i.lower().split())
                # check if we can find this one in the context
                match, word_idx_start, word_idx_end = better_subfinder(
                    words, answer_i.lower(), try_hard=True
                )
                if match:
                    break  # break inner
        if match:
            assert word_idx_start != -1 and word_idx_end != -1
            extracted_answer = " ".join(words[word_idx_start:word_idx_end + 1])
        else:
            word_idx_start = -1
            word_idx_end = -1
            extracted_answer = ""
        ## end index is inclusive
        if word_idx_start != -1:
            all_not_found = False
        processed_answers.append({
            "start_word_position": word_idx_start,
            "end_word_position": word_idx_end,
            "gold_answer": current_ans,
            "extracted_answer": extracted_answer})
    return processed_answers, all_not_found

def extract_start_end_index_v3(current_answers, words, threshold = 3):
    """
    Directly enumerate all the spans, really expensive, there is a threshold of tolerance
    :param current_answers:
    :param words:
    :param threshold: for example, the length of answer token is L, we search all the spans
                        whose length is within [L-threshold, L+threshold]
    :return:
    """
    ## extracting the start, end index
    processed_answers = []
    ## remove duplicates because of the case of multiple answers
    current_answers = list(set(current_answers))
    all_not_found = True
    for ans_index in range(len(current_answers)):
        current_ans = current_answers[ans_index]
        start_index, end_index, extracted_answer = get_answer_indices_by_enumeration(words, current_answers[ans_index], threshold)
        ans = current_ans.lower()
        extracted_answer = clean_text(extracted_answer)
        ans = clean_text(ans)
        dist = editdistance.eval(extracted_answer.replace(' ', ''),
                                 ans.replace(' ', '')) if extracted_answer != None else 1000
        if start_index == -1:
            start_index = -1
            end_index = -1
            extracted_answer = ""
        if dist > 5:
            start_index = -1
            end_index = -1
        if start_index == -1 or len(extracted_answer) > 150 or extracted_answer == "":
            start_index = -1
            end_index = -1
            extracted_answer = ""
        if start_index != -1:
            all_not_found = False
        processed_answers.append({
            "start_word_position": start_index,
            "end_word_position": end_index,
            "gold_answer": current_ans,
            "extracted_answer": extracted_answer})
    return processed_answers, all_not_found