import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import type { UploadResponse, HealthCheckResponse } from '@/types/api';

export const segmentationApi = createApi({
  reducerPath: 'segmentationApi',
  baseQuery: fetchBaseQuery({ 
    baseUrl: '/api',
    timeout: 30000, // 30 seconds timeout
  }),
  tagTypes: ['Health'],
  endpoints: (builder) => ({
    uploadImage: builder.mutation<UploadResponse, FormData>({
      query: (formData) => ({
        url: '/upload',
        method: 'POST',
        body: formData,
      }),
      // Retry failed requests up to 2 times with exponential backoff
      extraOptions: { maxRetries: 2 },
    }),
    healthCheck: builder.query<HealthCheckResponse, void>({
      query: () => '/health',
      providesTags: ['Health'],
      // Retry health check up to 3 times
      extraOptions: { maxRetries: 3 },
    }),
  }),
});

export const { useUploadImageMutation, useHealthCheckQuery } =
  segmentationApi;
