# IrisNeuralNet

[![CI](https://github.com/Stallker202/IrisNeuralNet/actions/workflows/ci.yml/badge.svg)](https://github.com/Stallker202/IrisNeuralNet/actions/workflows/ci.yml)

Полносвязная нейросеть, написанная с нуля на C# / .NET 8 без ML-фреймворков, и WinForms-лаборатория для экспериментов с ней. На этом движке выполнены четыре лабораторные работы, которые в методичках предлагается делать на Python + Keras: Iris, Sonar, Boston Housing и MNIST.

- **Ядро без зависимостей**: прямой и обратный проход, оптимизаторы, функции потерь, загрузка данных и сохранение модели написаны вручную; ни одного NuGet-пакета.
- **Быстро**: SIMD через `Vector<T>`, переиспользуемые буферы вместо аллокаций в цикле обучения, распараллеливание больших матричных произведений. Эпоха MNIST для сети 784 → 512 → 256 → 10 занимает несколько секунд.
- **Детерминированно**: при одинаковом seed результаты совпадают до бита, в том числе на Windows и Linux.
- **Проверено**: 129 тестов, включая сверку градиентов backprop с численными производными.

## Результаты лабораторных

Полные отчёты с графиками, таблицами и разбором лежат в [`reports/`](reports); их генерирует консольный режим (см. ниже).

| Лабораторная | Задача | Модель | Результат |
|---|---|---|---|
| [Iris](reports/iris/results.md) | многоклассовая классификация | 4 → 8 → 3, softmax, Adam | 100% на валидации (30 образцов) |
| [Sonar](reports/sonar/results.md) | бинарная классификация | 60 → 60 → 15 → 1, sigmoid, Adam | 85.7% на валидации, 84.9% по 5-fold CV |
| [Boston Housing](reports/boston/results.md) | регрессия | 13 → 64 → 64 → 1, linear, RMSprop | MAE 2.44 тыс. $ на тесте, 2.57 по 4-fold CV |
| [MNIST](reports/mnist/results.md) | классификация изображений | 784 → 512 → 256 → 10, RMSprop | 98.3% на тесте (10 000 изображений) |

В каждом отчёте пройдены пункты «Хода работы» из методички: формы данных, предобработка, `model.summary()`, обучение, `evaluate`, графики потерь и метрики, сравнение вариантов модели и разбор переобучения.

## Быстрый старт

Нужны .NET SDK 8 и Windows (для WinForms-лаборатории; ядро и тесты кроссплатформенные). Решение — `IrisNeuralNet.slnx` (Visual Studio 2022).

```bash
git clone https://github.com/Stallker202/IrisNeuralNet.git
cd IrisNeuralNet

dotnet test IrisNeuralNet.Tests                     # тесты
dotnet run --project IrisNeuralNet.Lab -c Release   # окно лаборатории
```

### Консольный режим: лабораторные целиком

```bash
dotnet run --project IrisNeuralNet.Lab -c Release -- --console all
dotnet run --project IrisNeuralNet.Lab -c Release -- --console mnist --image my_digit.png
dotnet run --project IrisNeuralNet.Lab -c Release -- --console iris sonar --quick
```

| Аргумент | Что делает |
|---|---|
| `iris`, `sonar`, `boston`, `mnist`, `all` | какие лабораторные прогнать (по умолчанию `iris`) |
| `--quick` | в 10 раз меньше эпох — быстрая проверка, что всё работает |
| `--out <папка>` | куда сохранить `results.md` и PNG (по умолчанию `./reports`) |
| `--image <файл>` | для MNIST: распознать своё изображение цифры (png, jpg, bmp) |
| `--no-charts` | не сохранять графики |

Полный прогон всех четырёх лаб занимает несколько минут; почти всё время уходит на MNIST.

### Окно лаборатории

Слева выбираются датасет, предобработка, архитектура скрытых слоёв (строкой, например `512, 256`), оптимизатор, learning rate (меняется на лету), эпохи и batch size. При выборе датасета подставляются значения из соответствующей методички.

| Вкладка | Что показывает |
|---|---|
| Обучение | живые графики потерь и метрики для train и validation с наложением прошлого прогона |
| Сеть и границы | веса и активации сети; проекция данных на две главные компоненты с границами решений (для регрессии — поверхность предсказаний) |
| Sweep | перебор «скрытые слои × нейроны» с тепловой картой качества и выбором лучшей архитектуры |
| Цифры (MNIST) | нарисуй цифру мышью или открой картинку — сеть показывает вероятности классов прямо во время обучения |

## Использование как библиотеки

```csharp
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Optimizers;

// Данные: загрузка, разбиение 80/20, z-score по обучающей части, one-hot
Dataset iris = IrisDataLoader.Load("iris.csv");
var (train, val) = DatasetSplitter.Split(iris, validationFraction: 0.2f, seed: 42);
FeatureNormalizer normalizer = FeatureNormalizer.Fit(train.Features, train.FeatureCount, train.RowCount);
float[] trainX = normalizer.Transform(train.Features, train.RowCount);
float[] valX = normalizer.Transform(val.Features, val.RowCount);
float[] trainY = TrainingTask.MultiClass.EncodeTargets(train.Labels, train.ClassCount);
float[] valY = TrainingTask.MultiClass.EncodeTargets(val.Labels, val.ClassCount);

// Модель: 4 → 8 (ReLU) → 3 (softmax), Adam, categorical cross-entropy
NeuralNetwork network = NetworkFactory.Create(4, new[] { 8 }, TrainingTask.MultiClass, classCount: 3, seed: 7);
Console.WriteLine(ModelSummary.Format(network));

var history = new TrainingHistory();
var trainer = new Trainer(network, new AdamOptimizer(network, learningRate: 0.01f), TrainingTask.MultiClass, history);
trainer.Fit(trainX, trainY, valX, valY, new TrainingOptions(Epochs: 150, BatchSize: 10, Seed: 7));

Evaluation result = trainer.Evaluate(valX, valY);
Console.WriteLine($"val loss {result.Loss:F4}, val accuracy {result.Metric:P1}");

// Модель сохраняется вместе с нормализатором, поэтому на вход подаются сырые признаки
ModelFile.Save(network, normalizer, "iris.nn");
LoadedModel model = ModelFile.Load("iris.nn");
float[] sample = model.Normalizer!.Transform(new[] { 5.1f, 3.5f, 1.4f, 0.2f }, rowCount: 1);
Console.WriteLine(IrisDataLoader.ClassNames[model.Network.PredictClass(sample)]);
```

Тип задачи (`TrainingTask`) определяет выходной слой, функцию потерь, метрику и кодирование меток:

| Задача | Выход | Потери | Метрика | Метки |
|---|---|---|---|---|
| `TrainingTask.MultiClass` | softmax, K нейронов | categorical cross-entropy | accuracy | one-hot |
| `TrainingTask.Binary` | sigmoid, 1 нейрон | binary cross-entropy | accuracy (p ≥ 0.5) | 0 / 1 |
| `TrainingTask.Regression` | linear, 1 нейрон | MSE | MAE | значение |

Оптимизаторы: `SgdOptimizer`, `SgdMomentumOptimizer`, `RmsPropOptimizer`, `AdamOptimizer` (у RMSprop и Adam параметры по умолчанию как в Keras); создание по имени — `OptimizerFactory`.

## Структура

```
IrisNeuralNet/                 ядро (net8.0, без зависимостей)
├── MathCore/                  SIMD-операции, параллельные матричные произведения, активации, PCA
├── Core/                      слои, сеть, типы задач, функции потерь, метрики, Trainer, формат модели
├── Optimizers/                SGD, SGD + momentum, RMSprop, Adam
├── Data/                      Dataset, разбиения и K-Fold, загрузчики CSV и MNIST (IDX), нормализация,
│                              приведение своих картинок цифр к виду MNIST
└── Diagnostics/               метрики эпох, история обучения, сводка модели, наблюдатели
IrisNeuralNet.Lab/             лаборатория (net8.0-windows, WinForms)
├── Experiments/               каталог датасетов, подготовка данных, сборка эксперимента, загрузка MNIST
├── Scenarios/                 консольные сценарии четырёх лабораторных и общий анализ результатов
├── Reporting/                 таблицы, форматирование, интерфейс рендера графиков
└── *.cs                       окно, панели и визуализации; iris.csv, sonar.csv, boston.csv
IrisNeuralNet.Tests/           xUnit, 129 тестов
IrisNeuralNet.Benchmarks/      BenchmarkDotNet: матричные произведения, слой MNIST-размера, обучение
reports/                       результаты лабораторных (results.md + PNG)
```

Зависимости строго в одну сторону: Lab, Tests и Benchmarks ссылаются на ядро, ядро не знает ни о ком. Код сценариев не использует System.Drawing, поэтому консольный режим работает на любой ОС (графики сохраняются только под Windows).

## Данные

| Датасет | Источник | Как попадает в проект |
|---|---|---|
| Iris | UCI, 150 × 4, 3 класса | `IrisNeuralNet.Lab/iris.csv` |
| Sonar | UCI, 208 × 60, 2 класса | `IrisNeuralNet.Lab/sonar.csv` |
| Boston Housing | 506 × 13, цена в тыс. $ | `IrisNeuralNet.Lab/boston.csv` |
| MNIST | 60 000 + 10 000 изображений 28 × 28 | скачивается при первом запуске в `%LOCALAPPDATA%\IrisNeuralNet\datasets\mnist` и проверяется по SHA-256; без интернета — положить четыре файла `*-ubyte.gz` в папку `mnist` рядом с exe |

Разбиение train/validation случайное и детерминированное по seed. Размер обучающей части считается как в Keras, поэтому для Boston получается ровно 404 / 102. В отличие от `validation_split` в Keras, строки перемешиваются: иначе в валидацию отсортированных `iris.csv` и `sonar.csv` попал бы один класс.

## Как это устроено

- **Память.** Матрицы хранятся построчно в плоских `float[]`, операции принимают `Span<float>`. Рабочие буферы только растут и переиспользуются, поэтому после первой эпохи цикл обучения не выделяет память (кроме служебных объектов распараллеливания на больших матрицах).
- **Матричное умножение** идёт в порядке i → k → j: внутренняя операция — векторное `dest += row × k` по непрерывной строке. Начиная с 2²² умножений строки результата раздаются по ядрам; каждый элемент считает один поток, поэтому результат не зависит от числа ядер.
- **Обратный проход.** dW = Xᵀ·δ, dX = δ·Wᵀ; первый слой dX не считает. Softmax и sigmoid численно устойчивы, а их связки с cross-entropy дают канонический градиент p − y (проверено тестами).
- **Обучение и UI.** Обучение идёт в фоновом потоке; метрики передаются в окно через кольцевой буфер, а визуализации читают сеть отдельным проходом без кэшей обучения и не мешают ему. Остановка срабатывает перед следующим батчем.
- **Формат модели** — собственный бинарный: сигнатура `NNET`, версия, слои (размеры, активация, веса, смещения) и нормализатор признаков. При загрузке проверяются размеры и целостность файла.

## Тесты, CI и бенчмарки

- `dotnet test IrisNeuralNet.Tests` — 129 тестов: математика и SIMD-хвосты, параллельный путь против эталона, градиенты против конечных разностей, функции потерь, оптимизаторы, тренер (оценка кусками, отмена, early stopping), загрузчики CSV и IDX, формат модели, предобработка изображений.
- [GitHub Actions](.github/workflows/ci.yml) на `windows-latest` на каждый push: сборка, тесты и быстрый прогон всех сценариев. Коммит с `[labs]` в сообщении (или ручной запуск) прогоняет лабораторные полностью и коммитит обновлённую папку `reports/`.
- `dotnet run -c Release --project IrisNeuralNet.Benchmarks` — сравнение SIMD-умножения с наивным, слой MNIST-размера с dX и без него, 20 эпох обучения; `MemoryDiagnoser` показывает выделения памяти.
