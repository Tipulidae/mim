import pytest

from projects.patient_history.util import select_multihot_columns


@pytest.fixture
def columns():
    return [
        'SV_ICD_I109',
        'SV_ICD_I252',
        'SV_ICD_E119',
        'SV_ICD_I489',
        'SV_ICD_I509',
        'OV_ICD_N18',
        'OV_ICD_I109',
        'OV_ICD_H30-H36',
        'OV_ICD_H25-H28',
        'OV_ICD_M79',
        'SV_OP_AF021',
        'SV_OP_AF037',
        'SV_OP_AF063',
        'SV_OP_AF020',
        'SV_OP_DT016',
        'SV_OP_FNG05',
        'OV_OP_XS100',
        'OV_OP_DR016',
        'OV_OP_AL003',
        'OV_OP_DT026',
        'ATC_N02BE',
        'ATC_A02BC',
        'ATC_B01AC06',
        'ATC_C07AB02',
        'ATC_N05CF',
        'ATC_N05BA',
        'Alias',
        'date',
        'admission_index',
        'admission_date'
    ]


class TestSelectMultihotColumns:
    def test_empty_sources_selects_both_sv_and_ov(self, columns):
        selected = select_multihot_columns(
            columns, diagnoses=-1, interventions=-1, meds=-1)

        expected = [
            'SV_ICD_I109',
            'SV_ICD_I252',
            'SV_ICD_E119',
            'SV_ICD_I489',
            'SV_ICD_I509',
            'OV_ICD_N18',
            'OV_ICD_I109',
            'OV_ICD_H30-H36',
            'OV_ICD_H25-H28',
            'OV_ICD_M79',
            'SV_OP_AF021',
            'SV_OP_AF037',
            'SV_OP_AF063',
            'SV_OP_AF020',
            'SV_OP_DT016',
            'SV_OP_FNG05',
            'OV_OP_XS100',
            'OV_OP_DR016',
            'OV_OP_AL003',
            'OV_OP_DT026',
            'ATC_N02BE',
            'ATC_A02BC',
            'ATC_B01AC06',
            'ATC_C07AB02',
            'ATC_N05CF',
            'ATC_N05BA',
        ]

        assert expected == selected

    def test_select_only_sv(self, columns):
        selected = select_multihot_columns(
            columns, source='sv', diagnoses=-1, interventions=-1)

        expected = [
            'SV_ICD_I109',
            'SV_ICD_I252',
            'SV_ICD_E119',
            'SV_ICD_I489',
            'SV_ICD_I509',
            'SV_OP_AF021',
            'SV_OP_AF037',
            'SV_OP_AF063',
            'SV_OP_AF020',
            'SV_OP_DT016',
            'SV_OP_FNG05',
        ]

        assert expected == selected

    def test_select_only_the_first_few(self, columns):
        selected = select_multihot_columns(
            columns, source='ov', diagnoses=2, interventions=3, meds=1)

        expected = [
            'OV_ICD_N18',
            'OV_ICD_I109',
            'OV_OP_XS100',
            'OV_OP_DR016',
            'OV_OP_AL003',
            'ATC_N02BE',
        ]

        assert expected == selected
