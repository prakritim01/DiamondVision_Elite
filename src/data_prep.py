import pandas as pd

def prepare_surat_data(input_path='data/diamonds.csv', output_path='data/surat_diamonds.csv'):
    df = pd.read_csv(input_path)
    # Convert USD to INR and apply 15% Surat wholesale discount
    df['surat_price_inr'] = (df['price'] * 83.5) * 0.85
    # Clean and save
    final_df = df[['carat', 'cut', 'color', 'clarity', 'surat_price_inr']]
    final_df.to_csv(output_path, index=False)
    return "✅ Surat dataset prepared."

if __name__ == "__main__":
    print(prepare_surat_data())