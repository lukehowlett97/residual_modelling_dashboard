import numpy as np

class RollingStatistics:
    def __init__(self, logger = None):
        self.logger = logger
        pass
    
    def _log(self, message):
        if self.logger:
            self.logger.write_log(message)
        else:
            pass
    
    def calculate(self, df, config):
        
        # Calculate rolling statistics
        for col in config['columns_to_process']:
            df[f'{col}-diff'] = df[col].diff()
            df[f'{col}-rolling_mean'] = df[col].rolling(window=config['rolling_window']).mean()
            df[f'{col}-rolling_std'] = df[col].rolling(window=config['rolling_window']).std()
            df[f'{col}-sg_filter'] = df[col].rolling(window=config['rolling_window']).apply(
                lambda x: np.polyfit(range(config['rolling_window']), x, config['poly_order'])[0]
                if len(x) == config['rolling_window'] else np.nan
            )
            
        return df