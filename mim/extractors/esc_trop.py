from mim.massage.esc_trop import make_ed_table, make_troponin_table
from mim.extractors.extractor import Data, Container, ECGData, Extractor, \
    DataProvider, SingleContainerLinearSplitProvider


class EscTrop(Extractor):
    def get_data_provider(self, dp_kwargs) -> DataProvider:
        ed = make_ed_table()
        tnt = make_troponin_table()
        ed = ed.join(tnt).reset_index()
        # ed['days_since_last_ecg'] = (ed.ecg_date - ed.old_ecg_date
        #                              ).dt.total_seconds() // (24 * 3600)
        ed.sex = ed.sex.apply(lambda x: 1 if x == 'M' else 0)

        ecg_path = '/mnt/air-crypt/air-crypt-esc-trop/axel/ecg.hdf5'

        mode = self.features['ecg_mode']

        x_dict = {}
        if 'index' in self.features['ecgs']:
            x_dict['ecg'] = ECGData(
                ecg_path,
                mode=mode,
                index=ed.ecg_id.astype(int).values
            )
        if 'old' in self.features['ecgs']:
            x_dict['old_ecg'] = ECGData(
                ecg_path,
                mode=mode,
                index=ed.old_ecg_id.astype(int).values
            )
        if 'features' in self.features:
            x_dict['features'] = Data(ed[self.features['features']].values)

        data = Container(
            {
                'x': Container.from_dict(x_dict),
                'y': Data(ed.mace_30_days.astype(int).values)
            },
            index=ed.index
        )

        return SingleContainerLinearSplitProvider(data, **dp_kwargs)
