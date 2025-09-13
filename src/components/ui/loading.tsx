import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const spinnerVariants = cva("animate-spin text-foreground/50", {
  variants: {
    size: {
      sm: "h-4 w-4",
      default: "h-5 w-5",
      lg: "h-6 w-6",
      xl: "h-8 w-8",
      "2xl": "h-10 w-10",
    },
    variant: {
      default: "text-foreground/50",
      primary: "text-primary",
      secondary: "text-secondary-foreground",
      success: "text-emerald-500",
      warning: "text-amber-500",
      danger: "text-destructive",
    },
  },
  defaultVariants: {
    size: "default",
    variant: "default",
  },
})

interface SpinnerProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof spinnerVariants> {}

export function Spinner({ className, size, variant, ...props }: SpinnerProps) {
  return (
    <div
      role="status"
      className={cn("inline-block", className)}
      {...props}
    >
      <svg
        className={cn(spinnerVariants({ size, variant }))}
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
      <span className="sr-only">Loading...</span>
    </div>
  )
}

const dotPulseVariants = cva("h-2 w-2 rounded-full bg-current", {
  variants: {
    size: {
      sm: "h-1.5 w-1.5",
      default: "h-2 w-2",
      lg: "h-3 w-3",
    },
    variant: {
      default: "bg-foreground/50",
      primary: "bg-primary",
      secondary: "bg-secondary-foreground",
      success: "bg-emerald-500",
      warning: "bg-amber-500",
      danger: "bg-destructive",
    },
  },
  defaultVariants: {
    size: "default",
    variant: "default",
  },
})

export function DotPulse({
  className,
  size,
  variant,
  ...props
}: React.HTMLAttributes<HTMLDivElement> &
  VariantProps<typeof dotPulseVariants>) {
  return (
    <div
      className={cn("flex items-center justify-center space-x-1.5", className)}
      {...props}
    >
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className={cn(
            dotPulseVariants({ size, variant }),
            "animate-bounce"
          )}
          style={{
            animationDelay: `${i * 0.15}s`,
            animationDuration: "1s",
            animationIterationCount: "infinite",
          }}
        />
      ))}
    </div>
  )
}

interface LoadingOverlayProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof spinnerVariants> {
  message?: string
  showSpinner?: boolean
  overlayClassName?: string
}

export function LoadingOverlay({
  className,
  size = "lg",
  variant = "primary",
  message,
  showSpinner = true,
  overlayClassName,
  children,
  ...props
}: LoadingOverlayProps) {
  return (
    <div className={cn("relative", className)} {...props}>
      {children}
      <div
        className={cn(
          "absolute inset-0 flex flex-col items-center justify-center gap-3 bg-background/80 backdrop-blur-sm",
          overlayClassName
        )}
      >
        {showSpinner && <Spinner size={size} variant={variant} />}
        {message && (
          <p className="text-sm font-medium text-foreground/80">{message}</p>
        )}
      </div>
    </div>
  )
}

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  isLoaded?: boolean
  fadeInDuration?: number
}

export function Skeleton({
  className,
  isLoaded = false,
  fadeInDuration = 0.2,
  children,
  ...props
}: SkeletonProps) {
  if (isLoaded) {
    return (
      <div
        className={cn("animate-fade-in", className)}
        style={{ animationDuration: `${fadeInDuration}s` }}
        {...props}
      >
        {children}
      </div>
    )
  }

  return (
    <div
      className={cn(
        "animate-pulse rounded-md bg-muted",
        !children && "h-4 w-full",
        className
      )}
      {...props}
    >
      {children && (
        <div className="invisible opacity-0">{children}</div>
      )}
    </div>
  )
}

interface SkeletonTextProps extends React.HTMLAttributes<HTMLDivElement> {
  lines?: number
  isLoaded?: boolean
  spacing?: number
  lineHeight?: number
}

export function SkeletonText({
  className,
  lines = 3,
  isLoaded = false,
  spacing = 8,
  lineHeight = 16,
  ...props
}: SkeletonTextProps) {
  if (isLoaded) {
    return <div className={className} {...props} />
  }

  return (
    <div className={cn("space-y-2", className)} {...props}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          style={{
            height: `${lineHeight}px`,
            width: i === lines - 1 && lines > 1 ? "80%" : "100%",
            marginTop: i > 0 ? `${spacing}px` : 0,
          }}
        />
      ))}
    </div>
  )
}

interface ProgressIndicatorProps
  extends React.HTMLAttributes<HTMLDivElement> {
  value: number
  max?: number
  showValue?: boolean
  variant?: "default" | "primary" | "success" | "warning" | "danger"
  size?: "sm" | "default" | "lg"
  isIndeterminate?: boolean
}

export function ProgressIndicator({
  value,
  max = 100,
  showValue = false,
  variant = "primary",
  size = "default",
  isIndeterminate = false,
  className,
  ...props
}: ProgressIndicatorProps) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100))
  const height = {
    sm: "h-1.5",
    default: "h-2",
    lg: "h-3",
  }[size]

  const variantClasses = {
    default: "bg-primary/20",
    primary: "bg-primary/20",
    success: "bg-emerald-500/20",
    warning: "bg-amber-500/20",
    danger: "bg-destructive/20",
  }

  const indicatorClasses = {
    default: "bg-primary",
    primary: "bg-primary",
    success: "bg-emerald-500",
    warning: "bg-amber-500",
    danger: "bg-destructive",
  }

  return (
    <div className={cn("w-full", className)} {...props}>
      <div className="flex items-center justify-between gap-2">
        {showValue && (
          <span className="text-xs font-medium text-muted-foreground">
            {isIndeterminate ? "Processing..." : `${Math.round(percentage)}%`}
          </span>
        )}
      </div>
      <div
        className={cn(
          "mt-1 w-full overflow-hidden rounded-full",
          variantClasses[variant],
          height
        )}
      >
        <div
          className={cn(
            "h-full rounded-full transition-all duration-300 ease-in-out",
            indicatorClasses[variant],
            isIndeterminate ? "w-full animate-indeterminate" : ""
          )}
          style={{
            width: isIndeterminate ? undefined : `${percentage}%`,
          }}
        />
      </div>
    </div>
  )
}

interface LoadingButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  isLoading?: boolean
  loadingText?: string
  loadingIndicator?: React.ReactNode
  variant?: "default" | "primary" | "secondary" | "outline" | "ghost" | "link"
  size?: "default" | "sm" | "lg" | "icon"
}

export function LoadingButton({
  children,
  isLoading = false,
  loadingText,
  loadingIndicator,
  disabled,
  className,
  variant = "default",
  size = "default",
  ...props
}: LoadingButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
        {
          "h-9 px-4 py-2": size === "default",
          "h-8 rounded-md px-3 text-xs": size === "sm",
          "h-10 rounded-md px-8": size === "lg",
          "h-9 w-9": size === "icon",
          "bg-primary text-primary-foreground hover:bg-primary/90":
            variant === "default",
          "bg-background text-foreground hover:bg-gray-200 hover:text-gray-900 border border-input":
            variant === "outline",
          "bg-secondary text-secondary-foreground hover:bg-secondary/80":
            variant === "secondary",
          "hover:bg-gray-200 hover:text-gray-900": variant === "ghost",
          "text-primary underline-offset-4 hover:underline": variant === "link",
        },
        className
      )}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <>
          {loadingIndicator || <Spinner size="sm" className="mr-2" />}
          {loadingText || "Loading..."}
        </>
      ) : (
        children
      )}
    </button>
  )
}

// Add global styles for animations
if (typeof document !== "undefined") {
  const style = document.createElement("style")
  style.textContent = `
    @keyframes indeterminate {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }
    .animate-indeterminate {
      animation: indeterminate 1.5s infinite ease-in-out;
    }
  `
  document.head.appendChild(style)
}
