import { RawData } from "@/types";
import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { useSearchParams } from "react-router";

type RemoteDataQueryOptions = Omit<
  UseQueryOptions<RawData>,
  "queryKey" | "queryFn"
>;

export const useRemoteData = (queryOptions: RemoteDataQueryOptions) => {
  const [searchParams] = useSearchParams();
  const remoteApi = searchParams.get("remote_api");

  return useQuery({
    queryKey: ["history"],
    queryFn: async () => {
      if (!remoteApi) throw new Error("No remote api provided");
      const res = await fetch(remoteApi);
      return (await res.json()) as RawData;
    },
    enabled: !!remoteApi && !queryOptions.enabled,
    ...queryOptions,
  });
};
