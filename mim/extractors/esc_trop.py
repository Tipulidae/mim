from mim.massage.esc_trop import make_ed_table, make_troponin_table
from mim.extractors.extractor import Data, Container, ECGData


class EscTrop:
    def __init__(self, specification):
        self.specification = specification

    def get_data(self):
        ed = make_ed_table()
        tnt = make_troponin_table()
        ed = ed.join(tnt).reset_index()

        ecg_path = '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5'

        mode = self.specification['features']['ecg']

        data = Container(
            {
                'x': ECGData(
                    ecg_path,
                    mode=mode,
                    index=ed.ecg_id.astype(int).values
                ),
                'y': Data(ed.mace_30_days.astype(int).values)
            },
            index=ed.index
        )

        return data
