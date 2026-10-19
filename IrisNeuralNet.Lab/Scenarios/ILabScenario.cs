namespace IrisNeuralNet.Lab.Scenarios;

/// <summary>Практическая часть одной лабораторной работы, воспроизведённая на движке IrisNeuralNet.</summary>
internal interface ILabScenario
{
    /// <summary>Ключ командной строки: iris, sonar, boston, mnist.</summary>
    string Key { get; }

    string Title { get; }

    void Run(ScenarioContext context);
}
