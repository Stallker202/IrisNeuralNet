# IrisNeuralNet

[![CI](https://github.com/Stallker202/IrisNeuralNet/actions/workflows/ci.yml/badge.svg)](https://github.com/Stallker202/IrisNeuralNet/actions/workflows/ci.yml)

Полносвязная нейросеть, написанная с нуля на C# / .NET 8 без ML-фреймворков, и веб-лаборатория для экспериментов с ней: работает в браузере на Windows, macOS и Linux. На этом движке выполнены четыре лабораторные работы.

- **Ядро без зависимостей**: прямой и обратный проход, оптимизаторы, функции потерь, загрузка данных и сохранение модели написаны вручную; ни одного NuGet-пакета.
- **Быстро**: SIMD через `Vector<T>`, переиспользуемые буферы вместо аллокаций в цикле обучения, распараллеливание больших матричных произведений. Эпоха MNIST для сети 784 → 512 → 256 → 10 занимает несколько секунд.
- **Детерминированно**: при одинаковом seed прогон на той же машине повторяется до бита. На другом процессоре или ОС возможны расхождения в последних знаках (разная ширина SIMD и математическая библиотека), на MNIST — до десятых процента точности.
- **Проверено**: 164 теста, включая сверку градиентов backprop с численными производными и сквозные тесты HTTP-API.

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

### Готовая сборка (без установки .NET)

Скачай архив под свою систему со страницы [Releases](https://github.com/Stallker202/IrisNeuralNet/releases), распакуй и запусти `IrisNeuralNet.Lab` (`IrisNeuralNet.Lab.exe` на Windows) — откроется браузер с лабораторией.

| Система | Архив |
|---|---|
| macOS, Apple Silicon (M1–M4) | `IrisNeuralNet-osx-arm64.tar.gz` |
| macOS, Intel | `IrisNeuralNet-osx-x64.tar.gz` |
| Windows | `IrisNeuralNet-win-x64.zip` |
| Linux | `IrisNeuralNet-linux-x64.tar.gz` |

На macOS сборка не подписана сертификатом Apple, поэтому один раз сними карантин:

```bash
xattr -dr com.apple.quarantine IrisNeuralNet-osx-arm64
./IrisNeuralNet-osx-arm64/IrisNeuralNet.Lab
```

### Из исходников

Нужен [.NET SDK 8](https://dotnet.microsoft.com/download) на любой ОС.

```bash
git clone https://github.com/Stallker202/IrisNeuralNet.git
cd IrisNeuralNet

dotnet run --project IrisNeuralNet.Lab -c Release   # веб-лаборатория, браузер откроется сам
dotnet test IrisNeuralNet.Tests                     # тесты
```

Лаборатория слушает только `http://127.0.0.1:5187` — из сети она не видна. Порт 5000 не используется: на macOS его занимает AirPlay. Флаги: `--port <номер>` (0 — любой свободный), `--no-browser`. В Visual Studio, Rider и VS Code проект запускается как обычное консольное приложение.

### Веб-лаборатория

Слева выбираются датасет, предобработка, архитектура скрытых слоёв (строкой, например `512, 256`), оптимизатор, learning rate (меняется на лету), эпохи и batch size. При выборе датасета подставляются значения из соответствующей методички.

| Вкладка | Что показывает |
|---|---|
| Обучение | живые графики потерь и метрики для train и validation с наложением прошлого прогона; при наведении — значения в эпохе |
| Сеть и границы | веса и активации сети; проекция данных на две главные компоненты с границами решений (для регрессии — поверхность предсказаний) |
| Sweep | перебор «скрытые слои × нейроны» с тепловой картой качества и выбором лучшей архитектуры |
| Цифры (MNIST) | нарисуй цифру мышью, открой картинку (любой формат, который понимает браузер) или возьми тестовую — сеть показывает вероятности классов прямо во время обучения |

У каждого графика есть кнопка **PNG** — для вставки в отчёт. Перезагрузка страницы не сбрасывает идущее обучение, а несколько вкладок браузера показывают одну и ту же лабораторию.

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
| `--out <папка>` | куда сохранить `results.md` и графики (по умолчанию `./reports`) |
| `--image <файл>` | для MNIST: распознать своё изображение цифры (PNG) |
| `--no-charts` | не сохранять графики |

Графики сохраняются в SVG — GitHub и Word показывают их как есть. Полный прогон всех четырёх лаб занимает несколько минут; почти всё время уходит на MNIST.

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
IrisNeuralNet.Lab/             лаборатория (net8.0 + ASP.NET Core из состава .NET, без NuGet)
├── Experiments/               каталог датасетов, подготовка данных, сборка эксперимента, загрузка MNIST
├── Training/                  фоновое обучение, sweep, история прогонов, read-only проба сети
├── Rendering/                 SVG-отрисовка графиков, топологии, границ решений, sweep — для страницы и отчётов
├── Imaging/                   PNG без зависимостей: превью цифр и --image
├── Web/                       состояние лаборатории на сервере и HTTP-API
├── Scenarios/                 консольные сценарии четырёх лабораторных и общий анализ результатов
├── Reporting/                 таблицы, форматирование, интерфейс сохранения графиков
├── wwwroot/                   страница: HTML, CSS и ~700 строк JavaScript без сборщиков
└── iris.csv, sonar.csv, boston.csv
IrisNeuralNet.Tests/           xUnit, 164 теста
IrisNeuralNet.Benchmarks/      BenchmarkDotNet: матричные произведения, слой MNIST-размера, обучение
reports/                       результаты лабораторных (results.md + SVG)
```

Зависимости строго в одну сторону: Lab, Tests и Benchmarks ссылаются на ядро, ядро не знает ни о ком.

## Данные

| Датасет | Источник | Как попадает в проект |
|---|---|---|
| Iris | UCI, 150 × 4, 3 класса | `IrisNeuralNet.Lab/iris.csv` |
| Sonar | UCI, 208 × 60, 2 класса | `IrisNeuralNet.Lab/sonar.csv` |
| Boston Housing | 506 × 13, цена в тыс. $ | `IrisNeuralNet.Lab/boston.csv` |
| MNIST | 60 000 + 10 000 изображений 28 × 28 | скачивается при первом выборе в кэш (`%LOCALAPPDATA%\IrisNeuralNet` на Windows, `~/Library/Application Support/IrisNeuralNet` на macOS, `~/.local/share/IrisNeuralNet` на Linux) и проверяется по SHA-256; без интернета — положить четыре файла `*-ubyte.gz` в папку `mnist` рядом с исполняемым файлом |

Разбиение train/validation случайное и детерминированное по seed. Размер обучающей части считается как в Keras, поэтому для Boston получается ровно 404 / 102. В отличие от `validation_split` в Keras, строки перемешиваются: иначе в валидацию отсортированных `iris.csv` и `sonar.csv` попал бы один класс.

## Как это устроено

- **Память.** Матрицы хранятся построчно в плоских `float[]`, операции принимают `Span<float>`. Рабочие буферы только растут и переиспользуются, поэтому после первой эпохи цикл обучения не выделяет память (кроме служебных объектов распараллеливания на больших матрицах).
- **Матричное умножение** идёт в порядке i → k → j: внутренняя операция — векторное `dest += row × k` по непрерывной строке. Начиная с 2²² умножений строки результата раздаются по ядрам; каждый элемент считает один поток, поэтому результат не зависит от числа ядер.
- **Обратный проход.** dW = Xᵀ·δ, dX = δ·Wᵀ; первый слой dX не считает. Softmax и sigmoid численно устойчивы, а их связки с cross-entropy дают канонический градиент p − y (проверено тестами).
- **Обучение и интерфейс.** Обучение идёт в фоновом потоке сервера; метрики собираются в кольцевой буфер, а визуализации читают сеть отдельным проходом без кэшей обучения и не мешают ему. Остановка срабатывает перед следующим батчем.
- **Отрисовка на сервере.** Всё, что раньше рисовал WinForms через System.Drawing, теперь рисует C# в SVG (`Rendering/`), в той же палитре и раскладке. Страница опрашивает состояние 5 раз в секунду и вставляет готовые SVG, а JavaScript отвечает только за контролы, подсказку при наведении, холст для цифр и экспорт в PNG. Тот же код в светлой теме рисует графики отчётов — одна реализация на оба случая. Цвета страницы тоже берутся из C# (`/theme.css` генерируется из `Theme`).
- **Формат модели** — собственный бинарный: сигнатура `NNET`, версия, слои (размеры, активация, веса, смещения) и нормализатор признаков. При загрузке проверяются размеры и целостность файла.

## Тесты, CI и бенчмарки

- `dotnet test IrisNeuralNet.Tests` — 164 теста: математика и SIMD-хвосты, параллельный путь против эталона, градиенты против конечных разностей, функции потерь, оптимизаторы, тренер (оценка кусками, отмена, early stopping), загрузчики CSV и IDX, формат модели, предобработка изображений, PNG-кодек (все фильтры и типы цвета), SVG-отрисовка, состояние лаборатории, sweep и HTTP-API на настоящем Kestrel.
- [GitHub Actions](.github/workflows/ci.yml) на каждый push: сборка, тесты и быстрый прогон всех сценариев на **Windows, macOS и Linux**, затем самодостаточные архивы под четыре платформы. Тег `vX.Y.Z` публикует их в Releases. Коммит с `[labs]` в сообщении (или ручной запуск) прогоняет лабораторные полностью и коммитит обновлённую папку `reports/`.
- `dotnet run -c Release --project IrisNeuralNet.Benchmarks` — сравнение SIMD-умножения с наивным, слой MNIST-размера с dX и без него, 20 эпох обучения; `MemoryDiagnoser` показывает выделения памяти.
