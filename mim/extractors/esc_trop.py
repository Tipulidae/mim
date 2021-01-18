from mim.massage.esc_trop import make_ed_table, make_troponin_table
from mim.extractors.extractor import Data, Container, ECGData


class EscTrop:
    def __init__(self, specification):
        self.specification = specification

    def get_data(self):
        ed = make_ed_table()
        tnt = make_troponin_table()
        ed = ed.join(tnt).reset_index()
        # ed['days_since_last_ecg'] = (ed.ecg_date - ed.old_ecg_date
        #                              ).dt.total_seconds() // (24 * 3600)
        ed.sex = ed.sex.apply(lambda x: 1 if x == 'M' else 0)

        ecg_path = '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5'

        mode = self.specification['features']['ecg_mode']

        data = Container(
            {
                'x': Container(
                    {
                        'ecg': ECGData(
                            ecg_path,
                            mode=mode,
                            index=ed.ecg_id.astype(int).values
                        ),
                        'old_ecg': ECGData(
                            ecg_path,
                            mode=mode,
                            index=ed.old_ecg_id.astype(int).values
                        ),
                        'features': Data(ed[['age', 'sex']].values)
                    }
                ),
                'y': Data(ed.mace_30_days.astype(int).values)
            },
            index=ed.index
        )

        return data
