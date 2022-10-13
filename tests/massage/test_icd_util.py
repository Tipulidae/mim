from massage.icd_util import round_icd_to_chapter


class TestRoundICDToChapter:
    def test_section_inputs(self):
        expected = {
            "A00-A09": "I",
            "B15-B19": "I",
            "B90-B94": "I",
            "B99-B99": "I",
            "C00-C75": "II",
            "D80-D89": "III",
            "F80-F89": "V",
            "G70-G73": "VI",
            "I10-I15": "IX",
            "I95-I99": "IX",
            "P80-P83": "XVI"
        }
        for section, chapter in expected.items():
            assert round_icd_to_chapter(section) == chapter

    def test_bad_inputs(self):
        bad_icds = ["\t\t\t", "9", "296W", "ZZZ"]
        for bad_icd in bad_icds:
            assert round_icd_to_chapter(bad_icd) == 'XXX'
