import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, RadioButtons
from sklearn.datasets import fetch_california_housing


def get_datasets():
    print("Przygotowywanie zestawów danych...")
    datasets = {}

    secret_degree = np.random.randint(1, 8)
    print(f"\n[SEKRET DLA PREZENTERA] Wygenerowano bazowy wielomian stopnia: {secret_degree}\n")
    
    x_mystery = np.linspace(-3, 3, 200)
    y_mystery = np.zeros_like(x_mystery)

    for i in range(secret_degree + 1):
        coef = np.random.uniform(-3, 3)
        y_mystery += coef * (x_mystery ** i)
        
    # Dodajemy szum Gaussa, żeby ukryć idealną linię (szum proporcjonalny do rozrzutu danych)
    noise_level = np.std(y_mystery) * 0.2
    y_mystery += np.random.randn(200) * noise_level

    datasets['Krzywa'] = {
        'x': x_mystery, 
        'y': y_mystery,
        'title': '(Wielomian z szumem)',
        'xlabel': 'Cecha X', 'ylabel': 'Wartość Y (z szumem)'
    }

    california = fetch_california_housing()
    datasets['Kalifornia'] = {
        'x': california.data[:400, 0],
        'y': california.target[:400],
        'title': 'Zależność ceny od zarobków (Kalifornia)',
        'xlabel': 'Średnie zarobki', 'ylabel': 'Cena domu'
    }

    x_stock = np.arange(200)
    y_stock = 100 + np.cumsum(np.random.randn(200) * 2)
    datasets['Giełda'] = {
        'x': x_stock, 'y': y_stock,
        'title': 'Symulacja ceny akcji',
        'xlabel': 'Dni', 'ylabel': 'Cena akcji'
    }

    return datasets

def create_polynomial_features(x, degree):
    m = len(x)
    X = np.ones((m, 1))
    for d in range(1, degree + 1):
        x_poly = np.power(x, d).reshape(-1, 1)
        X = np.column_stack((X, x_poly))
    return X

def standardize_features(X):
    X_norm = np.copy(X)
    if X.shape[1] > 1:
        means = np.mean(X[:, 1:], axis=0)
        stds = np.std(X[:, 1:], axis=0)
        stds[stds == 0] = 1 
        X_norm[:, 1:] = (X[:, 1:] - means) / stds
    else:
        means, stds = np.array([]), np.array([])
    return X_norm, means, stds

def gradient_descent(X, y, learning_rate=0.1, epochs=1000):
    m = len(y)
    theta = np.zeros(X.shape[1])
    loss_history = []
    
    for _ in range(epochs):
        predictions = X.dot(theta)
        errors = predictions - y
        loss = (1 / (2 * m)) * np.sum(errors**2)
        
        if np.isnan(loss) or np.isinf(loss):
            return theta, loss_history, True 
            
        loss_history.append(loss)
        
        gradients = (1 / m) * X.T.dot(errors)
        theta -= learning_rate * gradients
        
    return theta, loss_history, False


class PolynomialApp:
    def __init__(self, datasets):
        self.datasets = datasets
        self.current_dataset = 'Krzywa' 
        self.epochs = 1000
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        self.fig.subplots_adjust(bottom=0.3) 
        
        # Widget: Przełącznik zbiorów danych
        ax_radio = self.fig.add_axes([0.02, 0.05, 0.18, 0.15])
        ax_radio.set_title("Wybierz zbiór danych:")
        self.radio = RadioButtons(ax_radio, list(self.datasets.keys()), active=0)
        
        # Widget: Stopień wielomianu
        ax_box_deg = self.fig.add_axes([0.25, 0.15, 0.08, 0.05])
        self.text_box_deg = TextBox(ax_box_deg, 'Stopień: ', initial='1')
        
        # Widget: Krok uczenia (Learning Rate)
        ax_box_lr = self.fig.add_axes([0.25, 0.05, 0.08, 0.05])
        self.text_box_lr = TextBox(ax_box_lr, 'Krok (LR): ', initial='0.1')
        
        # Widget: Przycisk
        ax_button = self.fig.add_axes([0.38, 0.1, 0.1, 0.05])
        self.button = Button(ax_button, 'Przelicz', hovercolor='0.975')
        
        # Widget: Okno wyników
        ax_mse = self.fig.add_axes([0.52, 0.1, 0.45, 0.05])
        ax_mse.axis('off')
        self.mse_text = ax_mse.text(0.0, 0.5, 'Gotowy.', fontsize=11, 
                                    verticalalignment='center',
                                    bbox=dict(facecolor='lightyellow', edgecolor='black', boxstyle='round,pad=0.5'))
        
        # Zdarzenia
        self.button.on_clicked(self.calculate_and_plot)
        self.text_box_deg.on_submit(self.calculate_and_plot)
        self.text_box_lr.on_submit(self.calculate_and_plot)
        self.radio.on_clicked(self.change_dataset)
        
        self.calculate_and_plot(None)
        
    def change_dataset(self, label):
        self.current_dataset = label
        self.calculate_and_plot(None)
        
    def calculate_and_plot(self, event):
        try:
            degree = int(self.text_box_deg.text)
            lr = float(self.text_box_lr.text)
            if degree < 1 or lr <= 0:
                raise ValueError
        except ValueError:
            self.mse_text.set_text("Błąd: Podaj poprawne liczby (Stopień > 0, Krok > 0)!")
            self.fig.canvas.draw_idle()
            return
            
        data = self.datasets[self.current_dataset]
        x = data['x']
        y = data['y']
        
        X_poly = create_polynomial_features(x, degree)
        X_norm, means, stds = standardize_features(X_poly)
        
        theta, loss_history, exploded = gradient_descent(X_norm, y, learning_rate=lr, epochs=self.epochs)
        
        self.ax1.clear()
        self.ax2.clear()
        
        if exploded:
            self.ax1.text(0.5, 0.5, 'BŁĄD: EKSPLODUJĄCY GRADIENT!\nZmniejsz Krok uczenia (LR).', 
                          horizontalalignment='center', verticalalignment='center', 
                          fontsize=15, color='red', transform=self.ax1.transAxes)
            self.mse_text.set_text(f"Zbiór: {self.current_dataset} | Wynik: NaN (Zmniejsz parametr Krok!)")
            self.fig.canvas.draw_idle()
            return

        predictions = X_norm.dot(theta)
        mse = np.mean((predictions - y)**2)
        rmse = np.sqrt(mse)
        if self.current_dataset == 'Giełda':
            self.ax1.plot(x, y, alpha=0.6, color='gray', linestyle='dashed', marker='o', markersize=3, label='Dane rzeczywiste')
        else:
            self.ax1.scatter(x, y, alpha=0.5, color='blue', label='Dane rzeczywiste')
        
        x_line = np.linspace(min(x), max(x), 100)
        X_line_poly = create_polynomial_features(x_line, degree)
        X_line_norm = np.copy(X_line_poly)
        if degree > 0 and len(means) > 0:
            X_line_norm[:, 1:] = (X_line_poly[:, 1:] - means) / stds
            
        y_line = X_line_norm.dot(theta)
        self.ax1.plot(x_line, y_line, color='red', linewidth=3, label=f'Model st. {degree}')
        
        self.ax1.set_title(data['title'])
        self.ax1.set_xlabel(data['xlabel'])
        self.ax1.set_ylabel(data['ylabel'])
        self.ax1.legend()
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        
        self.ax2.plot(range(len(loss_history)), loss_history, color='green', linewidth=2)
        self.ax2.set_title('Krzywa uczenia (Spadek MSE)')
        self.ax2.set_xlabel('Epoki')
        self.ax2.set_ylabel('Funkcja straty J(theta)')
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        
        self.mse_text.set_text(f"Zbiór: {self.current_dataset} | MSE: {mse:.2f} | Średni błąd (RMSE): {rmse:.2f}")
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    datasets = get_datasets()
    print("Uruchamianie aplikacji")
    app = PolynomialApp(datasets)
    plt.show()