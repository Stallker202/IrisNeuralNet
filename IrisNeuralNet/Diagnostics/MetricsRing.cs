using System;
using System.Collections.Generic;

/// Кольцевой буфер метрик эпох: производитель — поток обучения, потребитель — поток UI.
/// Синхронизация — один короткий lock (раз в эпоху / раз в кадр):
/// для диагностического канала это надёжнее и дешевле lock-free кольца.

namespace IrisNeuralNet.Diagnostics
{
    public sealed class MetricsRing : ITrainingObserver
    {
        private readonly object _gate = new();
        private readonly EpochMetrics[] _items;
        private readonly int _mask;
        private int _head;
        private int _count;

        public MetricsRing(int capacity = 4096)
        {
            int rounded = 1;
            while (rounded < capacity) rounded <<= 1;
            _items = new EpochMetrics[rounded];
            _mask = rounded - 1;
        }

        public void OnEpochCompleted(in EpochMetrics metrics)
        {
            lock (_gate)
            {
                _items[_head] = metrics;
                _head = (_head + 1) & _mask;
                if (_count < _items.Length) _count++;
            }
        }

        public int Count
        {
            get { lock (_gate) return _count; }
        }

        /// <summary>Копирует метрики в список потребителя, от старых к новым.</summary>
        public void CopyTo(List<EpochMetrics> destination)
        {
            lock (_gate)
            {
                destination.Clear();
                int start = (_head - _count) & _mask;
                for (int i = 0; i < _count; i++)
                {
                    destination.Add(_items[(start + i) & _mask]);
                }
            }
        }

        public void Clear()
        {
            lock (_gate)
            {
                _head = 0;
                _count = 0;
            }
        }
    }
}
