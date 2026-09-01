import type { Metadata } from "next";

import { PRODUCT_NAME } from "@bos/card-schema";

import { CompareBoard } from "@/components/CompareBoard";
import { loadCompareData } from "@/lib/load";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: `Compare · ${PRODUCT_NAME}`,
};

type Props = {
  searchParams: Promise<{
    mode?: string | string[];
    season?: string | string[];
    ids?: string | string[];
  }>;
};

export default async function ComparePage({ searchParams }: Props) {
  const params = await searchParams;
  const data = await loadCompareData(params);
  return <CompareBoard {...data} />;
}
