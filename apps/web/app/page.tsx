import { Home } from "@/components/Home";
import { loadHomeData } from "@/lib/load";

export const dynamic = "force-dynamic";

export default async function Page() {
  const data = await loadHomeData();
  return <Home {...data} />;
}
