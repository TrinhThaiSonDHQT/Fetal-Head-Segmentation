import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import type { UploadResponse, HealthCheckResponse } from '@/types/api';

export const segmentationApi = createApi({
  reducerPath: 'segmentationApi',
  baseQuery: fetchBaseQuery({ baseUrl: '/api' }),
  tagTypes: ['Health'],
  endpoints: (builder) => ({
    uploadImage: builder.mutation<UploadResponse, FormData>({
      query: (formData) => ({
        url: '/upload',
        method: 'POST',
        body: formData,
      }),
    }),
    healthCheck: builder.query<HealthCheckResponse, void>({
      query: () => '/health',
      providesTags: ['Health'],
    }),
  }),
});

export const { useUploadImageMutation, useHealthCheckQuery } =
  segmentationApi;
