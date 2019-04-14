using System;
using System.Numerics;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;

public class FFT : IDisposable
{
	public enum ESize { N16 = 16, N32 = 32, N64 = 64, N128 = 128, N256 = 256, N512 = 512 }

	public struct TransformJob : IJob
	{
		public int N;

		[ReadOnly]
		public NativeArray<uint> reversed;

		[ReadOnly]
		public NativeArray<Complex> time;

		public NativeArray<Complex> data;

		private uint log2N;

		public void Execute() {
			log2N = (uint)Mathf.Log(N, 2);

			Complex[][] c = new Complex[2][];
			c[0] = new Complex[N];
			c[1] = new Complex[N];

			for(int mPrime = 0; mPrime < N; mPrime++) {
				Transform(ref c, 1, mPrime * N);
			}

			for(int nPrime = 0; nPrime < N; nPrime++) {
				Transform(ref c, N, nPrime);
			}
		}

		private void Transform(ref Complex[][] c, int stride, int offset) {
			uint which = 0;

			for(int i = 0; i < N; i++) {
				c[which][i] = data[(int)(reversed[i] * stride + offset)];
			}

			int loops = N >> 1;
			int size = 1 << 1;
			int sizeOver2 = 1;
			int w_ = 0;

			for(int i = 1; i <= log2N; i++) {
				which ^= 1;

				for(int j = 0; j < loops; j++) {
					for(int k = 0; k < sizeOver2; k++) {
						c[which][size * j + k] = c[which ^ 1][size * j + k] +
							c[which ^ 1][size * j + sizeOver2 + k] * time[(w_ * N) + k];
					}

					for(int k = sizeOver2; k < size; k++) {
						c[which][size * j + k] = c[which ^ 1][size * j - sizeOver2 + k] -
							c[which ^ 1][size * j + k] * time[(w_ * N) + (k - sizeOver2)];
					}
				}

				loops >>= 1;
				size <<= 1;
				sizeOver2 <<= 1;
				w_++;
			}

			for(int i = 0; i < N; i++) {
				data[i * stride + offset] = c[which][i];
			}
		}
	}

	public int N { get; private set; }
	public int Nplus1 { get { return N + 1; } }

	[ReadOnly]
	private NativeArray<uint> reversed;

	[ReadOnly]
	private NativeArray<Complex> time;

	private uint log2N = 0;
	private bool disposed = false;

	public FFT(ESize size) {
		N = (int)size;
		log2N = (uint)Mathf.Log(N, 2);

		reversed = new NativeArray<uint>(N, Allocator.Persistent);
		time = new NativeArray<Complex>(N * N, Allocator.Persistent);

		for(int i = 0; i < N; i++) {
			reversed[i] = ReverseBits((uint)i);
		}

		int pow2 = 1;

		for(int i = 0; i < log2N; i++) {
			for(int j = 0; j < pow2; j++) {
				float term = (Mathf.PI * 2) * j / (pow2 * 2);
				time[(i * N) + j] = new Complex(Mathf.Cos(term), Mathf.Sin(term));
			}

			pow2 *= 2;
		}
	}

	~FFT() {
		Dispose();
	}

	public void Dispose() {
		if(disposed)
			return;

		reversed.Dispose();
		time.Dispose();

		GC.SuppressFinalize(this);
		disposed = true;
	}

	private uint ReverseBits(uint i) {
		uint ri = 0;

		for(int j = 0; j < log2N; j++) {
			ri = (ri << 1) + (i & 1);
			i >>= 1;
		}

		return ri;
	}

	public JobHandle ScheduleTransform(ref NativeArray<Complex> data) {
		return new TransformJob {
			N = this.N,
			reversed = this.reversed,
			time = this.time,
			data = data
		}.Schedule();
	}
}
