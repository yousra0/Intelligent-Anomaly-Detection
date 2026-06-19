import Link from "next/link";
import { cn } from "@/lib/utils";

interface LogoProps {
  href?: string;
  size?: "sm" | "md" | "lg" | "xl";
  showWordmark?: boolean;
  className?: string;
}

const SIZES = {
  sm: { imgSize: 32, wordmark: "text-xs" },
  md: { imgSize: 40, wordmark: "text-sm" },
  lg: { imgSize: 56, wordmark: "text-base" },
  xl: { imgSize: 72, wordmark: "text-lg" },
};

export function Logo({ href, size = "md", showWordmark = true, className }: LogoProps) {
  const { imgSize, wordmark } = SIZES[size];

  const inner = (
    <div className={cn("flex items-center gap-2.5", className)}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="/png_logo.png"
        alt="PwC"
        width={imgSize}
        height={imgSize}
        className="shrink-0 object-contain"
      />
     
    </div>
  );

  if (href) {
    return (
      <Link href={href} className="flex items-center outline-none focus-visible:ring-2 focus-visible:ring-ring rounded">
        {inner}
      </Link>
    );
  }

  return inner;
}
