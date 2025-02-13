import ta
import optuna
import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from datetime import timedelta, datetime
from qtrader.environments.base import BaseMarketEnv
from qtrader.stateproviders import BaseSymbolStateProvider
from qtrader.rlflow.persistence import BasePersistenceProvider

optuna.logging.set_verbosity(optuna.logging.CRITICAL)


class IndicatorModelSymbolStateProvider(BaseSymbolStateProvider):

    def __init__(self, stock_env: BaseMarketEnv, symbol: str,
                 days_ago: int = 28 * 12 * 3,
                 num_trials: int = 15 * 2,
                 refresh_rate_days: int = 35,
                 **kwargs):
        super(IndicatorModelSymbolStateProvider, self).__init__(stock_env, symbol, **kwargs)
        self.days_ago = days_ago
        self.num_trials = num_trials
        self.refresh_rate_days = refresh_rate_days
        self.n_estimators = 25
        self.p_days_ahead = 5
        self.params = None
        self.model = None

        self.skip_save = True

    @staticmethod
    def create_dataset(df, lag=0, window=14, window_slow=22, window_fast=12, mode='train', p_days_ahead=5, **kwargs):
        df_t = df.copy()

        # BollingerBands
        bb = ta.volatility.BollingerBands(close=df.close, window=window, window_dev=2)
        df_t.loc[:, 'bb_bbh'] = 1 - bb.bollinger_hband() / df.close
        df_t.loc[:, 'bb_bbl'] = 1 - bb.bollinger_lband() / df.close
        df_t.loc[:, 'bb_bbw'] = bb.bollinger_wband() / df.close
        df_t.loc[:, 'bb_bbp'] = bb.bollinger_pband()

        # RSI
        rsi = ta.momentum.RSIIndicator(close=df.close, window=window)
        srsi = ta.momentum.StochRSIIndicator(close=df.close, window=window)
        df_t.loc[:, 'rsi'] = rsi.rsi()
        df_t.loc[:, 'srsi'] = srsi.stochrsi()

        # MACD
        macd = ta.trend.MACD(close=df.close, window_slow=window_slow, window_fast=window_fast)
        df_t.loc[:, 'macd'] = macd.macd() / macd.macd().std()
        df_t.loc[:, 'macd_diff'] = macd.macd_diff() / macd.macd_diff().std()
        df_t.loc[:, 'macd_signal'] = macd.macd_signal() / macd.macd_signal().std()

        # ATR
        atr = ta.volatility.AverageTrueRange(high=df.high, low=df.low, close=df.close, window=window)
        df_t.loc[:, 'atr'] = 100 * atr.average_true_range() / df.close

        # StochasticOscillator
        so = ta.momentum.StochasticOscillator(high=df.high, low=df.low, close=df.close, window=window)
        df_t.loc[:, 'so'] = so.stoch()
        df_t.loc[:, 'sos'] = so.stoch_signal()

        # PSARIndicator
        sar = ta.trend.PSARIndicator(high=df.high, low=df.low, close=df.close)
        df_t.loc[:, 'sar'] = sar.psar()

        # MFIIndicator
        if df.volume.sum() > 0:
            mfi = ta.volume.MFIIndicator(high=df.high, low=df.low, close=df.close, volume=df.volume, window=window)
            df_t.loc[:, 'mfi'] = mfi.money_flow_index()

            # VolumePriceTrendIndicator
            vpt = ta.volume.VolumePriceTrendIndicator(close=df.close, volume=df.volume)
            df_t.loc[:, 'vpt'] = vpt.volume_price_trend()

        # TRIX
        trix = ta.trend.TRIXIndicator(close=df.close, window=window)
        df_t.loc[:, 'trix'] = trix.trix()

        df_t = df_t.drop(['open', 'high', 'low', 'close', 'volume', 'datetime'], axis=1)
        for c in df_t.columns:
            for l in range(lag):
                df_t.loc[:, f'{c}_lag_{l + 1}'] = df_t[c].shift(l + 1)

        if mode == 'train':
            df_t.loc[:, 'target'] = (df['close'].shift(-p_days_ahead) - df['close']).map(
                lambda c: None if np.isnan(c) else 1 if c >= 0 else 0
            )
            df_t = df_t.dropna()
            return df_t.drop('target', axis=1), df_t['target']

        return df_t, None

    @staticmethod
    def objective(trial, df, n_estimators, p_days_ahead):
        window_slow = trial.suggest_int("window_slow", 8, 52)
        window = trial.suggest_int("window", 6, window_slow - 1)
        window_fast = trial.suggest_int("window_fast", 3, window - 1)

        try:
            X, y = IndicatorModelSymbolStateProvider.create_dataset(df,
                                                                    lag=trial.suggest_int("lag", 0, 5),
                                                                    window=window, window_slow=window_slow,
                                                                    window_fast=window_fast,
                                                                    mode='train',
                                                                    p_days_ahead=p_days_ahead
                                                                    )

            if X.shape[0] < 200:
                print(f'X.shape: {X.shape}; y.shape: {y.shape}')
                return 0

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion='entropy',
                max_depth=trial.suggest_int("max_depth", 1, 8),
                max_features=trial.suggest_int("max_features", int(np.sqrt(len(X.columns)) / 2), len(X.columns)),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 2 ** 6, step=2),
                random_state=42
            )

            n_splits = 3
            auc = 0
            cv = sklearn.model_selection.TimeSeriesSplit(n_splits=n_splits, gap=7, test_size=7 * 5)
            for train_index, test_index in cv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model.fit(X_train, y_train)
                y_p = model.predict_proba(X_test)[:, 1]
                auc += sklearn.metrics.roc_auc_score(y_test, y_p)

            trial.set_user_attr("auc", auc / n_splits)
            return auc / n_splits

        except Exception as e:
            print(f"EXCEPTION during trial: {e}; {trial.params}")
            return 0

    def load_config(self, pprovider: BasePersistenceProvider):
        try:
            self.params = pprovider.load_dict(f"IndicatorModelStateProvider-{self.symbol}")
            self.model = pprovider.load_obj(f"IndicatorModelStateProvider-{self.symbol}-Model")

        except Exception as e:
            self.params = {}

    def save_config(self, pprovider: BasePersistenceProvider):
        if not self.skip_save:
            pprovider.persist_dict(f"IndicatorModelStateProvider-{self.symbol}", self.params)
            pprovider.persist_obj(f"IndicatorModelStateProvider-{self.symbol}-Model", self.model)

    def _model_ts(self):
        if self.params and 'ts_model' in self.params:
            return datetime.fromisoformat(self.params['ts_model']).replace(tzinfo=None)

    def provide(self):
        cdt = self.stock_env.get_current_market_datetime()
        data = self.stock_env.get_ohlcv(self.symbol, dt_from=cdt - timedelta(days=self.days_ago), dt_to=cdt)
        print(f"Indicator Model[{self.symbol}]: {data.shape}")
        if len(data) < 600:
            return {'model_ind': None}

        m_ts = self._model_ts()  # model timestamp
        # not self.params -> first run, full retrain
        # (cdt - m_ts).days < 0 -> current ts is less than model ts, end of tr loop, full retrain
        # (cdt - m_ts).days > self.refresh_rate_days -> model is old, full retrain
        if not self.params or \
                (cdt - m_ts).days < 0 or \
                (cdt - m_ts).days > self.refresh_rate_days:

            # find optimal hyperparams
            study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=42))

            if self.params and 'best_trial' in self.params:  # if best_trial params exists add it to new study
                study.enqueue_trial(self.params['best_trial'])

            study.optimize(
                lambda t: IndicatorModelSymbolStateProvider.objective(t, data, self.n_estimators, self.p_days_ahead),
                n_trials=self.num_trials)

            self.params.update({'best_trial': study.best_params})
            self.params.update({'eval': study.best_trial.user_attrs})
            self.params['ts_model'] = cdt.isoformat()

            # build a model on all data
            X, y = IndicatorModelSymbolStateProvider.create_dataset(data,  # .iloc[:-self.p_days_ahead],
                                                                    mode='train',
                                                                    p_days_ahead=self.p_days_ahead,
                                                                    **self.params['best_trial'])
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion='entropy',
                max_depth=self.params['best_trial']['max_depth'],
                max_features=self.params['best_trial']['max_features'],
                min_samples_split=self.params['best_trial']['min_samples_split'],
                random_state=42
            )
            self.model.fit(X, y)
            self.skip_save = False
            print(f"Retrained: IndicatorModelStateProvider-{self.symbol}: {self.params}")

        try:
            # predict on latest data
            n_pred = 10
            X, _ = IndicatorModelSymbolStateProvider.create_dataset(data,
                                                                    mode='predict',
                                                                    **self.params['best_trial'])
            X = X.iloc[-n_pred:]
            data['target'] = (data['close'].shift(-self.p_days_ahead) - data['close']).map(
                lambda c: None if np.isnan(c) else 1 if c >= 0 else 0
            )
            data_last_n = data.iloc[-n_pred:][['close', 'target']]
            data_last_n['p'] = self.model.predict_proba(X)[:, 1]

            return {
                'model_ind': {
                    'params': self.params,
                    'preds': data_last_n.to_dict(orient='list')
                }
            }

        except Exception as e:
            print(f"EXCEPTION: IndicatorModelSymbolStateProvider[{self.symbol}]: {e}")
            return {'model_ind': None}
